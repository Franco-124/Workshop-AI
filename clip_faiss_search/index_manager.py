import json

import faiss
import numpy as np

from config import Config


class FAISSIndexManager:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._index: faiss.Index | None = None
        self._id_to_path: dict[int, str] = {}

    def build(self, embeddings: np.ndarray, image_paths: list[str]) -> None:
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._id_to_path = {i: path for i, path in enumerate(image_paths)}

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Índice no inicializado. Llama a build() o load() primero.")
        scores, indices = self._index.search(query_vector.astype(np.float32), top_k)
        return [
            (self._id_to_path[int(idx)], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def save(self) -> None:
        faiss.write_index(self._index, str(self._config.index_path))
        self._config.metadata_path.write_text(
            json.dumps(self._id_to_path), encoding="utf-8"
        )

    def load(self) -> None:
        self._index = faiss.read_index(str(self._config.index_path))
        raw = json.loads(self._config.metadata_path.read_text(encoding="utf-8"))
        self._id_to_path = {int(k): v for k, v in raw.items()}
