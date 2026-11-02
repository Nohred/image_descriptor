# LLM local y embeddings

`src/local_llm/` centraliza la comunicación con Ollama para dos tareas:

- interpretar consultas en lenguaje natural como filtros JSON;
- convertir consultas y descripciones en vectores para similitud semántica.

La configuración se encuentra en [`config.py`](config.py):

| Parámetro | Valor actual |
|---|---|
| Endpoint | `http://127.0.0.1:11434` |
| Modelo de chat | `qwen2.5:3b` |
| Modelo de embeddings | `nomic-embed-text-v2-moe` |
| Dimensión esperada | `768` |
| Temperatura | `0.0` |
| Contexto | `4096` |
| Timeout | `120 s` |

## Componentes

- `client.py`: cliente Ollama y `chat_json`, que fuerza respuestas JSON y
  permite enviar un esquema de salida.
- `embedder.py`: `embed_texts` y `embed_query`. Procesa lotes de 16, valida la
  forma `(n, 768)`, rechaza vectores inválidos y normaliza cada vector.
  Utiliza prefijos distintos para documentos y consultas.
- `prompts.py`: genera el esquema y el prompt para extraer tipos, colores,
  exclusiones, intervalos horarios/fechas y texto semántico residual.

## Instalación y modelos

Instale Ollama según su plataforma, inicie el servicio y descargue los
modelos:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text-v2-moe
```

Instale el cliente Python en el entorno que ejecutará la aplicación:

```bash
python -m pip install ollama numpy
```

El endpoint puede cambiarse en `config.py` si Ollama corre en otra máquina o
puerto. El modelo de embeddings debe devolver exactamente 768 dimensiones,
porque el generador y el repositorio validan esa condición.

## Ejemplo

```python
from src.local_llm.client import chat_json
from src.local_llm.embedder import embed_query

response = chat_json(
    "Devuelve filtros de vehículos.",
    "autos rojos durante la noche",
    {"type": "object"},
)
vector = embed_query("vehículo con parabrisas roto")
```

En la práctica, estas funciones se invocan a través de
[`retrieval.service`](../retrieval/README.md), no desde la interfaz web
directamente.

