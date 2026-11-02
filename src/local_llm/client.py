import json
"""
This module provides a client for interacting with the Ollama API.
It allows sending chat messages to a specified model and receiving structured JSON responses.
The client is configured with parameters such as the model name, context size, temperature, and timeout.
"""


from ollama import Client

from .config import (
    OLLAMA_ENDPOINT,
    CHAT_MODEL,
    CHAT_CONTEXT,
    TEMPERATURE,
    TIMEOUT,
)

_client = Client(
    host=OLLAMA_ENDPOINT,
    timeout=TIMEOUT,
)


def chat_json(
    system: str,
    user: str,
    schema: dict | None = None,
) -> dict:
    if schema is not None:
        system = (
            f"{system}\n\n"
            "La respuesta debe cumplir este esquema JSON:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )

    response = _client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=schema if schema is not None else "json",
        options={
            "temperature": TEMPERATURE,
            "num_ctx": CHAT_CONTEXT,
        },
    )

    return json.loads(response.message.content)