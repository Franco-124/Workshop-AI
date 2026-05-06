from PIL import Image

from config import Config
from embedder import CLIPEmbedder
from index_manager import FAISSIndexManager


class ImageSearcher:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._embedder = CLIPEmbedder(config)
        self._index_manager = FAISSIndexManager(config)
        self._index_manager.load()
        print("Modelo e índice cargados correctamente")

    def search_by_text(self, query: str) -> list[tuple[str, float]]:
        vector = self._embedder.embed_text(query)
        return self._index_manager.search(vector, self._config.top_k)

    def search_by_image(self, image: Image.Image) -> list[tuple[str, float]]:
        vector = self._embedder.embed_images([image])
        return self._index_manager.search(vector, self._config.top_k)

    @property
    def top_k(self) -> int:
        return self._config.top_k
