# clip_faiss_search

Motor de búsqueda de imágenes por texto usando CLIP + FAISS.

## Stack

- Python 3.11
- `transformers` — modelo CLIP para generar embeddings de texto e imagen
- `faiss-cpu` — índice vectorial para búsqueda por similitud
- `torch` (CPU only) — backend de inferencia para CLIP
- `Pillow` — carga y preprocesamiento de imágenes
- `gradio` — UI web para búsqueda interactiva

## Convenciones

- **Clean Code**: nombres descriptivos, funciones pequeñas y cohesivas
- **SOLID**: cada clase tiene una única responsabilidad; depende de abstracciones, no de implementaciones concretas
- **Type hints** en todas las funciones (parámetros y retorno)
- **Dataclasses** con `frozen=True` para toda la configuración (inmutables por diseño)
- Sin comentarios obvios; solo comentarios que explican el *por qué*, no el *qué*

## Estructura y responsabilidades

| Archivo | Clase/Rol | Responsabilidad |
|---|---|---|
| `config.py` | `AppConfig` (dataclass frozen) | Parámetros centralizados: modelo, rutas, device, top-k |
| `embedder.py` | `CLIPEmbedder` | Genera embeddings de texto e imagen; nunca hace I/O |
| `index_manager.py` | `FAISSIndexManager` | Construye, persiste y carga el índice FAISS |
| `indexer.py` | script batch | Recorre `images/`, genera embeddings y construye el índice |
| `searcher.py` | `ImageSearcher` | Orquesta `CLIPEmbedder` + `FAISSIndexManager` para búsqueda |
| `main.py` | UI Gradio | Punto de entrada de la interfaz; importa solo `ImageSearcher` |
| `download_images.py` | script | Descarga 500 imágenes de COCO val2017 a `images/` |

## Regla importante

Cada archivo tiene **una sola responsabilidad**. Ningún módulo importa de `main.py`.
El flujo de dependencias es estrictamente: `main.py → searcher.py → embedder.py + index_manager.py ← indexer.py`.
