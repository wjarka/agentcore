import logging
import os
from typing import override

import httpx
from openai import AsyncOpenAI
from openai.types import (
    CreateEmbeddingResponse,
)

from agentcore.services import EmbeddingService

logger = logging.getLogger(__name__)


class DefaultEmbeddingService(EmbeddingService):
    openai_client: AsyncOpenAI
    jina_api_key: str | None

    def __init__(self, openai_client: AsyncOpenAI):
        """
        Initializes the EmbeddingService.

        Args:
            openai_client: An instance of langfuse.openai.AsyncOpenAI.
        """
        self.openai_client = openai_client
        self.jina_api_key = os.getenv("JINA_API_KEY")
        if not self.jina_api_key:
            logger.warning(
                "JINA_API_KEY environment variable not set. "
                + "Jina embeddings will not be available unless explicitly set"
                + "or the key is provided."
            )

    @override
    async def get_openai_embedding(
        self, text: str, model: str = "text-embedding-3-large"
    ) -> list[float]:
        """
        Creates an embedding for the given text using OpenAI.

        Args:
            text: The text to embed.
            model: The OpenAI model to use for embedding.

        Returns:
            A list of floats representing the embedding.
        """
        try:
            response: CreateEmbeddingResponse = (
                await self.openai_client.embeddings.create(
                    model=model,
                    input=text,
                )
            )
            return response.data[0].embedding
        except Exception as error:
            logger.error(
                "Error creating OpenAI embedding via EmbeddingService: %s",
                error,
                exc_info=True,
            )
            raise error

    @override
    async def get_jina_embedding(self, text: str) -> list[float]:
        """
        Creates an embedding for the given text using the Jina AI API.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        if not self.jina_api_key:
            logger.error("JINA_API_KEY must be set to use Jina embeddings.")
            raise ValueError("JINA_API_KEY must be set for Jina embeddings.")

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.jina_api_key}",
            }
            payload = {
                "model": "jina-embeddings-v3",
                "input": [text],
                "task": "text-matching",
                "dimensions": 1024,
                "late_chunking": False,
                "embedding_type": "float",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                )
                _ = response.raise_for_status()
                data = response.json()

                if (
                    data
                    and isinstance(data.get("data"), list)
                    and len(data["data"]) > 0
                    and isinstance(data["data"][0].get("embedding"), list)
                ):
                    return data["data"][0]["embedding"]
                else:
                    logger.error(f"Unexpected response structure from Jina API: {data}")
                    raise ValueError(
                        "Failed to parse embedding from Jina API response."
                    )

        except httpx.HTTPStatusError as http_err:
            logger.error(
                f"HTTP error occurred while creating Jina embedding: {http_err} - Response: {http_err.response.text}",
                exc_info=True,
            )
            raise http_err
        except httpx.RequestError as req_err:
            logger.error(
                f"Request error occurred while creating Jina embedding: {req_err}",
                exc_info=True,
            )
            raise req_err
        except ValueError as val_err:
            logger.error(str(val_err), exc_info=True)
            raise val_err
        except Exception as error:
            logger.error(
                "An unexpected error occurred creating Jina embedding: %s",
                error,
                exc_info=True,
            )
            raise error
