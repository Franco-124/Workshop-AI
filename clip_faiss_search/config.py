from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_id: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    images_dir: Path = Path("./images/val2017")
    index_path: Path = Path("./faiss.index")
    metadata_path: Path = Path("./metadata.json")
    top_k: int = 5
    batch_size: int = 32
