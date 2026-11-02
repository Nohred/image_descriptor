"""Aplicacion web local para consultar los descriptores existentes."""
from dataclasses import asdict
import json
import mimetypes
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
MAX_RESULTS = 100
IMAGE_DIR = PROJECT_ROOT / "VeRi" / "image_test"
WEB_DIR = PROJECT_ROOT / "web"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.schemas import DEFAULT_LIMIT, MODES, SearchResult  # noqa: E402
from retrieval.service import SearchService  # noqa: E402


class UnavailableSource:
    def __init__(self, label, error):
        self.label = label
        self.error = error

    def search(self, *args, **kwargs):
        raise RuntimeError(f"{self.label} no está disponible: {self.error}")


def build_service():
    fixed_error = None
    try:
        from retrieval.fixed_source import FixedSource
        fixed = FixedSource(image_dir=IMAGE_DIR)
    except Exception as error:
        fixed_error = str(error)
        fixed = UnavailableSource("La búsqueda fixed", fixed_error)

    try:
        from database.generated_features.repository import GeneratedSource
        generated = GeneratedSource(image_dir=IMAGE_DIR)
        generated_error = None
    except Exception as error:
        generated = None
        generated_error = str(error)

    try:
        from local_llm.client import chat_json
        chat_fn = chat_json
        chat_error = None
    except Exception as error:
        chat_error = str(error)

        def chat_fn(*args, **kwargs):
            raise RuntimeError(f"El cliente de consultas no está disponible: {chat_error}")

    service = SearchService(fixed, chat_fn, generated_source=generated)
    return service, {
        "fixed": fixed_error,
        "semantic": generated_error,
        "hybrid": generated_error or chat_error,
    }


SERVICE, STARTUP_ERRORS = build_service()


def safe_image_path(image_id):
    """Resuelve únicamente nombres que permanecen dentro de IMAGE_DIR."""
    root = IMAGE_DIR.resolve()
    candidate = (root / image_id).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("La ruta de imagen está fuera de la carpeta permitida.")
    if not candidate.is_file():
        raise FileNotFoundError("La imagen solicitada no existe.")
    return candidate


def result_payload(result: SearchResult):
    payload = asdict(result)
    for candidate in payload["candidates"]:
        try:
            safe_image_path(candidate["image_id"])
            candidate["image_url"] = "/images/" + candidate["image_id"]
        except (ValueError, FileNotFoundError) as error:
            candidate["image_url"] = None
            candidate["image_error"] = str(error)
    return payload


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type=None):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        if parsed.path.startswith("/images/"):
            try:
                image_id = unquote(parsed.path[len("/images/"):])
                image_path = safe_image_path(image_id)
            except (ValueError, FileNotFoundError) as error:
                return self._send_json(404, {"error": str(error)})
            return self._send_file(image_path, mimetypes.guess_type(image_path)[0])
        if parsed.path == "/api/status":
            return self._send_json(200, {
                "modes": list(MODES),
                "startup_errors": STARTUP_ERRORS,
            })
        return self._send_json(404, {"error": "Ruta no encontrada."})

    def do_POST(self):
        if urlparse(self.path).path != "/api/search":
            return self._send_json(404, {"error": "Ruta no encontrada."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            query = request.get("query")
            mode = request.get("mode", "fixed")
            limit = request.get("limit", DEFAULT_LIMIT)
            if not isinstance(query, str) or not query.strip():
                raise ValueError("La consulta no puede estar vacía.")
            if mode not in MODES:
                raise ValueError(f"Modo inválido. Opciones: {', '.join(MODES)}.")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RESULTS:
                raise ValueError(f"El límite debe ser un entero entre 1 y {MAX_RESULTS}.")
            result = SERVICE.search(query.strip(), mode=mode, limit=limit)
            return self._send_json(200, result_payload(result))
        except (json.JSONDecodeError, TypeError):
            return self._send_json(400, {"error": "El cuerpo debe ser JSON válido."})
        except ValueError as error:
            return self._send_json(400, {"error": str(error)})
        except (RuntimeError, ModuleNotFoundError, ConnectionError, OSError) as error:
            return self._send_json(503, {"error": f"Modo no disponible: {error}"})
        except Exception as error:
            return self._send_json(500, {"error": f"Error interno de búsqueda: {error}"})

    def log_message(self, format, *args):
        print(f"[web] {format % args}")


if __name__ == "__main__":
    print(f"Aplicación disponible en http://{HOST}:{PORT}")
    if any(STARTUP_ERRORS.values()):
        print("Modos con problemas al iniciar:", STARTUP_ERRORS)
    ThreadingHTTPServer((HOST, PORT), RequestHandler).serve_forever()
