from __future__ import annotations

import math
from typing import Any, override

import tiktoken
from tiktoken import Encoding

from agentcore.log import logger
from agentcore.models import Document, Headers, Metadata
from agentcore.services import TextService


class DefaultTextService(TextService):
    model_name: str
    encoding: Encoding | None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name if model_name else "gpt-4o"
        self.encoding = None

    def _initialize_tokenizer(self, model_override: str | None = None) -> None:
        if model_override and model_override != self.model_name:
            logger().info(
                f"Model override provided: '{model_override}'. Current model: '{self.model_name}'."
            )
            self.model_name = model_override
            self.encoding = None
            logger().info(
                f"Tokenizer model name updated to: {self.model_name}. Will re-initialize."
            )

        if self.encoding is None:
            logger().info(f"Initializing tokenizer for model: {self.model_name}")
            try:
                self.encoding = tiktoken.encoding_for_model(self.model_name)
            except Exception as e:
                logger().error(
                    f"Failed to initialize tokenizer for model {self.model_name}: {e}"
                )
                raise

    def _format_for_tokenization(self, text: str) -> str:
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant<|im_end|>"

    def _count_tokens(self, text: str) -> int:
        if self.encoding is None:
            logger().error("Tokenizer accessed before initialization.")
            raise RuntimeError(
                "Tokenizer not initialized. Call _initialize_tokenizer first."
            )
        formatted_text = self._format_for_tokenization(text)
        tokens = self.encoding.encode(formatted_text, allowed_special="all")
        return len(tokens)

    def _find_new_chunk_end(self, start: int, end: int) -> int:
        # Reduce end position to try to fit within token limit
        new_end = end - math.floor((end - start) / 10)  # Reduce by 10% each iteration
        if new_end <= start:
            new_end = start + 1  # Ensure at least one character is included
        return new_end

    def _validate_chunk(self, text: str, start: int, end: int, limit: int) -> bool:
        min_chunk_tokens = limit * 0.8
        chunk_text = text[start:end]
        tokens = self._count_tokens(chunk_text)
        return tokens <= limit and tokens >= min_chunk_tokens

    def _adjust_chunk_end(self, text: str, start: int, end: int, limit: int) -> int:
        # Find the next newline after current end position
        next_newline = text.find("\n", end)
        prev_newline = text.rfind("\n", 0, end)
        if next_newline != -1:
            end = next_newline + 1
            if self._validate_chunk(text, start, end, limit):
                logger().info(f"Adjusted chunk to next newline at position {end}")
                return end
        if prev_newline > start:
            end = prev_newline + 1
            if self._validate_chunk(text, start, end, limit):
                logger().info(f"Adjusted chunk to previous newline at position {end}")
                return end
        return end

    def _get_chunk(self, text: str, start: int, limit: int) -> tuple[str, int]:
        logger().info(f"Getting chunk starting at position {start} with limit {limit}")
        overhead = self._count_tokens(
            self._format_for_tokenization("")
        ) - self._count_tokens("")

        # Calculate initial end position, avoiding division by zero
        remaining_text = text[start:]
        remaining_tokens = self._count_tokens(remaining_text)

        if remaining_tokens == 0:
            # If no tokens remaining, return the rest of the text
            end = len(text)
        else:
            end = min(
                start + math.floor((len(text) - start) * limit / remaining_tokens),
                len(text),
            )

        chunk_text = text[start:end]
        tokens = self._count_tokens(chunk_text)

        # Add safety counter to prevent infinite loops
        max_iterations = 50
        iteration_count = 0

        while (
            tokens + overhead > limit
            and end > start
            and iteration_count < max_iterations
        ):
            logger().info(
                f"Chunk exceeds limit with {tokens + overhead} tokens, reducing size (iteration {iteration_count + 1})"
            )
            end = self._find_new_chunk_end(start, end)
            chunk_text = text[start:end]
            tokens = self._count_tokens(chunk_text)
            iteration_count += 1

        if iteration_count >= max_iterations:
            logger().warning(
                f"Reached maximum iterations ({max_iterations}) while trying to fit chunk within token limit. "
                + f"Using best available chunk with {tokens + overhead} tokens."
            )

        end = self._adjust_chunk_end(text, start, end, limit)

        chunk_text = text[start:end]
        tokens = self._count_tokens(chunk_text)
        logger().info(f"Final chunk end: {end}")
        return chunk_text, end

    def _extract_headers(self, text: str) -> Headers:
        import re

        headers = Headers()
        pattern = r"(^|\n)(#{1,6})\s+(.*)"
        matches = re.findall(pattern, text, re.MULTILINE)
        for match in matches:
            level = len(match[1])  # Count the hash characters
            title = match[2].strip()  # Extract and strip the title
            attr_name = "h" + str(level)
            current_list = getattr(headers, attr_name)
            if current_list is None:
                setattr(headers, attr_name, [title])
            else:
                current_list.append(title)
        return headers

    def _extract_urls_and_images(self, text: str) -> tuple[str, list[str], list[str]]:
        import re

        urls: list[str] = []
        images: list[str] = []
        url_to_index: dict[str, int] = {}
        image_to_index: dict[str, int] = {}

        def replace_image(match: re.Match[str]):
            alt_text = match.group(1)
            url = match.group(2)

            if url in image_to_index:
                # Use existing index for duplicate image
                index = image_to_index[url]
            else:
                # Add new image and assign index
                index = len(images)
                images.append(url)
                image_to_index[url] = index

            result = f"![{alt_text}]({{{{$img{index}}}}})"
            return result

        def replace_url(match: re.Match[str]):
            link_text = match.group(1)
            url = match.group(2)
            # Don't process image placeholders as regular URLs
            if url.startswith("{{$img"):
                return match.group(0)  # Return the original match unchanged

            if url in url_to_index:
                # Use existing index for duplicate URL
                index = url_to_index[url]
            else:
                # Add new URL and assign index
                index = len(urls)
                urls.append(url)
                url_to_index[url] = index

            result = f"[{link_text}]({{{{$url{index}}}}})"
            return result

        # First replace images (which have ! prefix)
        content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, text)

        # Then replace regular links
        content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace_url, content)

        return content, urls, images

    @override
    def document(
        self,
        text: str,
        additional_metadata: dict[str, Any] | None = None,
        model: str | None = None,  # User can specify a model for this specific document
    ) -> Document:
        self._initialize_tokenizer(
            model_override=model
        )  # Initialize/update tokenizer if model is specified

        tokens = self._count_tokens(text)
        headers = self._extract_headers(text)
        content, urls, images = self._extract_urls_and_images(text)

        doc_metadata_payload: dict[str, Any] = {
            "tokens": tokens,
            "headers": headers,
            "urls": urls,
            "images": images,
        }
        if additional_metadata:
            doc_metadata_payload.update(additional_metadata)

        return Document(text=content, metadata=Metadata(**doc_metadata_payload))

    @override
    def restore_placeholders(self, doc: Document) -> Document:
        import re

        restored_text = doc.text
        metadata = doc.metadata

        if metadata.images:
            for i, img_url in enumerate(metadata.images):
                # Placeholder format: ({{$img<index>}})
                placeholder_pattern = r"\(\{\{\$img" + str(i) + r"\}\}\)"
                replacement_string = f"({img_url})"
                restored_text = re.sub(
                    placeholder_pattern, replacement_string, restored_text
                )

        if metadata.urls:
            for i, url_val in enumerate(metadata.urls):
                # Placeholder format: ({{$url<index>}})
                placeholder_pattern = r"\(\{\{\$url" + str(i) + r"\}\}\)"
                replacement_string = f"({url_val})"
                restored_text = re.sub(
                    placeholder_pattern, replacement_string, restored_text
                )

        return Document(
            text=restored_text,
            # Create a copy of metadata to avoid modifying the original Document's metadata
            metadata=metadata.model_copy(deep=True),
        )

    @override
    def split(
        self, text: str, limit: int, additional_metadata: dict[str, Any] | None = None
    ) -> list[Document]:
        self._initialize_tokenizer()  # Ensure tokenizer is initialized with the service's current model

        chunks: list[Document] = []
        position = 0
        total_length = len(text)
        current_headers_accumulator = Headers()  # Accumulator for headers across chunks

        while position < total_length:
            logger().info(f"Splitting text starting at position {position}")
            chunk_text, chunk_end = self._get_chunk(text, position, limit)

            # Extract headers from this specific chunk to update the accumulator
            headers_in_chunk = self._extract_headers(chunk_text)
            self._update_current_headers(current_headers_accumulator, headers_in_chunk)

            # Call self.document() to process the chunk_text.
            # It will handle token counting, URL/image extraction, placeholder replacement,
            # and merging of `additional_metadata`.
            # We pass self.model_name to ensure document() uses the same tokenizer settings
            # without re-initializing if it's already set for the current model.
            doc_for_this_chunk = self.document(
                text=chunk_text,
                model=self.model_name,
                additional_metadata=additional_metadata,
            )

            # The doc_for_this_chunk.metadata initially contains headers extracted *only* from chunk_text.
            # We now override these with the accumulated current_headers_accumulator.
            # All other metadata fields (tokens, urls, images, and merged additional_metadata)
            # are correctly calculated by the document() method for the chunk_text.
            doc_for_this_chunk.metadata.headers = (
                current_headers_accumulator.model_copy(deep=True)
            )

            chunks.append(doc_for_this_chunk)  # Append the modified document

            logger().info(f"Chunk processed. New position: {chunk_end}")
            position = chunk_end
        logger().info(f"Split process completed. Total chunks: {len(chunks)}")
        return chunks

    def _update_current_headers(
        self, current_headers: Headers, headers_in_chunk: Headers
    ):
        for level in range(1, 7):
            attr_name = "h" + str(level)
            chunk_headers = getattr(headers_in_chunk, attr_name)
            if chunk_headers is not None:
                setattr(current_headers, attr_name, chunk_headers)
                self._clear_lower_levels(current_headers, level)

    def _clear_lower_levels(self, current_headers: Headers, level: int):
        for i in range(level + 1, 7):
            attr_name = "h" + str(i)
            setattr(current_headers, attr_name, None)
