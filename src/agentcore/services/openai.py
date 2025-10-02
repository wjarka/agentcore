import asyncio
import base64
import datetime
from collections.abc import AsyncIterable
from typing import (
    Any,
    Literal,
    TypedDict,
    overload,
    override,
)

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)
from pydantic import TypeAdapter

from agentcore.log import logger
from agentcore.models import Document
from agentcore.prompts.protocols import Prompt, SystemPrompt
from agentcore.services import EmbeddingService, LLMService, TextService
from agentcore.telemetry import Telemetry


class ImageProcessingResult(TypedDict):
    description: str
    source: str


class OpenAIService(LLMService):
    openai: AsyncOpenAI
    text_service: TextService
    embedding_service: EmbeddingService
    telemetry: Telemetry

    def __init__(
        self,
        embedding_service: EmbeddingService,
        text_service: TextService,
        telemetry: Telemetry,
    ) -> None:
        """
        Initializes the OpenAIService.

        Args:
            embedding_service: An instance of EmbeddingService.
            text_service: An instance of TextService.
        """
        self.openai = AsyncOpenAI()
        self.text_service = text_service
        self.embedding_service = embedding_service
        self.telemetry = telemetry

    @overload
    async def completion(
        self,
        *,
        user_prompt: str | ChatCompletionMessageParam | Prompt | None = ...,
        system_prompt: str | ChatCompletionMessageParam | SystemPrompt | None = ...,
        history: list[ChatCompletionMessageParam] | None = ...,
        model: str = ...,
        json_mode: bool = ...,
        max_tokens: int | None = ...,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> ChatCompletion: ...

    @overload
    async def completion(
        self,
        *,
        user_prompt: str | ChatCompletionMessageParam | Prompt | None = ...,
        system_prompt: str | ChatCompletionMessageParam | SystemPrompt | None = ...,
        history: list[ChatCompletionMessageParam] | None = ...,
        model: str = ...,
        stream: Literal[False] = False,
        json_mode: bool = ...,
        max_tokens: int | None = ...,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> ChatCompletion: ...

    @overload
    async def completion(
        self,
        *,
        user_prompt: str | ChatCompletionMessageParam | Prompt | None = ...,
        system_prompt: str | ChatCompletionMessageParam | SystemPrompt | None = ...,
        history: list[ChatCompletionMessageParam] | None = ...,
        model: str = ...,
        stream: Literal[True] = True,
        json_mode: bool = ...,
        max_tokens: int | None = ...,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> AsyncIterable[ChatCompletionChunk]: ...

    @override
    async def completion(
        self,
        *,
        user_prompt: str | ChatCompletionMessageParam | Prompt | None = None,
        system_prompt: str | ChatCompletionMessageParam | SystemPrompt | None = None,
        history: list[ChatCompletionMessageParam] | None = None,
        model: str = "gpt-4.1",
        stream: bool = False,
        json_mode: bool = False,
        max_tokens: int | None = None,
        cache_key: str | None = None,
        name: str | None = None,
    ) -> ChatCompletion | AsyncIterable[ChatCompletionChunk]:
        """
        Generates a chat completion using the OpenAI API.

        Args:
            messages: A list of messages comprising the conversation so far.
            model: ID of the model to use.
            stream: If True, stream back partial progress.
            json_mode: If True, enable JSON mode for the response.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            A ChatCompletion object or an AsyncIterable of ChatCompletionChunk if streaming.
        """
        try:
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt is not None:
                messages.append(await self._prompt_to_message(system_prompt))
                if isinstance(system_prompt, SystemPrompt):
                    model = model or system_prompt.suggested_model
                    json_mode = json_mode or system_prompt.json_mode
                    max_tokens = max_tokens or system_prompt.max_tokens
                if cache_key is None and isinstance(system_prompt, Prompt):
                    cache_key = system_prompt.cache_key
            if history is not None:
                messages.extend(history)

            if user_prompt is not None:
                messages.append(await self._prompt_to_message(user_prompt))
                if cache_key is None and isinstance(user_prompt, Prompt):
                    cache_key = user_prompt.cache_key

            params_for_create: dict[str, Any] = {
                "messages": messages,
                "model": model,
            }

            extra_body = {}
            if cache_key is not None:
                extra_body["prompt_cache_key"] = cache_key

            params_for_create["extra_body"] = extra_body

            if model not in (
                "o1-mini",
                "o1-preview",
                "o1-mini-20240718",
                "o1-vision-20240718",
            ):
                if max_tokens is not None:
                    params_for_create["max_tokens"] = max_tokens
                if json_mode:
                    params_for_create["response_format"] = {"type": "json_object"}
                else:
                    params_for_create["response_format"] = {"type": "text"}
            elif stream:
                logger().warning(
                    f"""Model {model} does not support streaming, max_tokens, or response_format options.
                    Proceeding with a non-streaming completion without these options."""
                )
                stream = False
            with self.telemetry.generation(
                name=name or "AI Generation",
                completion_start_time=datetime.datetime.now(tz=datetime.timezone.utc),
                input=messages,
                model=model,
                model_parameters={
                    "temperature": 1,
                    "top_p": 1,
                    "max_tokens": max_tokens or float("inf"),
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
            ) as generation:
                if not stream:
                    completion = await self.openai.chat.completions.create(
                        stream=stream, **params_for_create
                    )
                    generation.set_output(completion)
                    if completion.usage is not None:
                        generation.set_usage(completion.usage.model_dump())
                    return completion
                else:

                    async def _stream():
                        async for chunk in await self.openai.chat.completions.create(
                            stream=stream, **params_for_create
                        ):
                            generation.append_output(chunk.choices[0].delta.content)
                            if chunk.usage is not None:
                                generation.add_usage(chunk.usage.model_dump())
                            yield chunk

                    return _stream()

        except Exception as error:
            logger().error("Error in OpenAI completion: %s", error, exc_info=True)
            raise error

    async def _prompt_to_message(
        self, prompt: str | ChatCompletionMessageParam | Prompt
    ) -> ChatCompletionMessageParam:
        if isinstance(prompt, str):
            return {"role": "system", "content": prompt}
        elif isinstance(prompt, Prompt):
            return await prompt.to_message()
        adapter = TypeAdapter[ChatCompletionMessageParam](ChatCompletionMessageParam)
        _ = adapter.validate_python(prompt)
        return prompt

    async def describe_image(self, image_path: str) -> ImageProcessingResult:
        """
        Processes a single image using GPT-4 Vision and returns its description.

        Args:
            image_path: Path to the image file.

        Returns:
            An ImageProcessingResult containing the description and source path.
        """
        try:
            loop = asyncio.get_running_loop()
            with open(image_path, "rb") as f:
                image_bytes = await loop.run_in_executor(None, f.read)
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif image_path.lower().endswith(".gif"):
                mime_type = "image/gif"
            elif image_path.lower().endswith(".webp"):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Defaulting
                logger().warning(
                    f"Could not determine MIME type for {image_path}, defaulting to image/jpeg."
                )

            response = await self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
            )
            description = "No description available."
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                description = response.choices[0].message.content

            return {
                "description": description,
                "source": image_path,
            }
        except Exception as error:
            logger().error(
                f"Error processing image {image_path}: %s", error, exc_info=True
            )
            raise error

    async def describe_images(
        self, image_paths: list[str]
    ) -> list[ImageProcessingResult]:
        """
        Processes multiple images concurrently.

        Args:
            image_paths: A list of paths to image files.

        Returns:
            A list of ImageProcessingResult objects.
        """
        try:
            results = await asyncio.gather(
                *(self.describe_image(path) for path in image_paths)
            )
            return results
        except Exception as error:
            logger().error("Error processing multiple images: %s", error, exc_info=True)
            raise error

    async def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image using GPT-4 Vision.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text from the image.
        """
        try:
            with open(image_path, "rb") as f:
                image_bytes = await asyncio.to_thread(f.read)
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif image_path.lower().endswith(".gif"):
                mime_type = "image/gif"
            elif image_path.lower().endswith(".webp"):
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
                logger().warning(
                    f"Could not determine MIME type for {image_path}, defaulting to image/jpeg."
                )

            response = await self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please extract all text from this image. Return only the text content, preserving the original formatting and structure as much as possible.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2048,
            )

            extracted_text = ""
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                extracted_text = response.choices[0].message.content

            return extracted_text

        except Exception as error:
            logger().error(
                f"Error extracting text from image {image_path}: %s",
                error,
                exc_info=True,
            )
            raise error

    async def transcribe_buffer(
        self,
        audio_buffer: bytes,
        language: str = "en",
        prompt: str | None = None,
    ) -> str:
        """
        Transcribes an audio buffer using OpenAI Whisper.

        Args:
            audio_buffer: The audio data in bytes.
            language: The language of the audio. Defaults to 'en'.
            prompt: An optional prompt to guide the transcription.

        Returns:
            The transcribed text.
        """
        logger().info("Transcribing audio buffer...")
        try:
            file_tuple: tuple[str, bytes] = ("speech.ogg", audio_buffer)
            transcription_obj = await self.openai.audio.transcriptions.create(
                file=file_tuple,
                language=language,
                model="whisper-1",
                prompt=prompt if prompt is not None else "",
            )
            return transcription_obj.text
        except Exception as error:
            logger().error("Error in OpenAI transcription: %s", error, exc_info=True)
            raise error

    @override
    async def transcribe(
        self,
        audio_files: list[str],
        language: str = "en",
        prompt: str | None = None,
        file_name: str = "transcription.md",
    ) -> list[Document]:
        """
        Transcribes multiple audio files and processes them into Document objects.

        Args:
            audio_files: A list of paths to audio files.
            language: The language of the audio. Defaults to 'pl'.
            prompt: An optional prompt to guide the transcription.
            file_name: The name to assign to the metadata of the created documents.

        Returns:
            A list of Document objects.
        """
        logger().info("Transcribing multiple audio files...")

        async def process_file(file_path: str) -> Document:
            with open(file_path, "rb") as f:
                buffer = await asyncio.to_thread(f.read)
            transcription_text = await self.transcribe_buffer(
                buffer, language=language, prompt=prompt
            )
            doc = self.text_service.document(
                text=transcription_text,
                model="gpt-4o",
                additional_metadata={"source": file_path, "name": file_name},
            )
            return doc

        try:
            results = await asyncio.gather(*(process_file(fp) for fp in audio_files))
            return results
        except Exception as error:
            logger().error(
                "Error transcribing multiple audio files: %s", error, exc_info=True
            )
            raise error
