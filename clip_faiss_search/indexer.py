from pathlib import Path

import numpy as np
from PIL import Image

from config import Config
from embedder import CLIPEmbedder
from index_manager import FAISSIndexManager

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_paths(images_dir: Path) -> list[Path]:
    return [p for p in images_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]


def run_indexing(config: Config) -> None:
    image_paths = collect_image_paths(config.images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {config.images_dir}")

    total = len(image_paths)
    print(f"Indexando {total} imágenes...")

    embedder = CLIPEmbedder(config)
    index_manager = FAISSIndexManager(config)

    all_embeddings: list[np.ndarray] = []
    procesadas = 0

    for i in range(0, total, config.batch_size):
        batch_paths = image_paths[i : i + config.batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        embeddings = embedder.embed_images(batch_images)
        all_embeddings.append(embeddings)
        procesadas += len(batch_paths)
        print(f"  {procesadas}/{total} imágenes procesadas")

    matrix = np.vstack(all_embeddings)
    index_manager.build(matrix, [str(p) for p in image_paths])
    index_manager.save()

    print(f"Índice guardado en {config.index_path}")
    print(f"Metadata guardada en {config.metadata_path}")


if __name__ == "__main__":
    config = Config()
    run_indexing(config)
