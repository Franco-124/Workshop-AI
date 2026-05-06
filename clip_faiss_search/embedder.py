import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import Config


class CLIPEmbedder:
    def __init__(self, config: Config) -> None:
        self._processor: CLIPProcessor = CLIPProcessor.from_pretrained(config.model_id)
        self._model: CLIPModel = CLIPModel.from_pretrained(config.model_id)
        self._model.eval()

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        outputs = self._model.get_image_features(**inputs)
        print(f"Tipo de outputs: {type(outputs)}")
        print(f"Atributos disponibles: {dir(outputs)}")
        if hasattr(outputs, "image_embeds"):
            features = outputs.image_embeds
        elif hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs
        return self._l2_normalize(features.detach().numpy().astype(np.float32))

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        outputs = self._model.get_text_features(**inputs)
        if hasattr(outputs, "text_embeds"):
            features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs
        return self._l2_normalize(features.detach().numpy().astype(np.float32))

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.clip(norms, 1e-10, None)).astype(np.float32)
