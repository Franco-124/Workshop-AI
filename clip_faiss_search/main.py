import gradio as gr
from PIL import Image

from config import Config
from searcher import ImageSearcher

config = Config()
searcher = ImageSearcher(config)


def search_by_text(query: str) -> list[str]:
    if not query or not query.strip():
        return []
    results = searcher.search_by_text(query)
    return [path for path, score in results]


def search_by_image(image: Image.Image) -> list[str]:
    if image is None:
        return []
    results = searcher.search_by_image(image)
    return [path for path, score in results]


with gr.Blocks(title="Búsqueda visual con CLIP + FAISS") as demo:
    with gr.Tab("Buscar por texto"):
        textbox = gr.Textbox(label="Describe una imagen", placeholder="a dog playing in snow")
        text_btn = gr.Button(value="Buscar")
        gallery = gr.Gallery(label="Resultados", columns=5)
        text_btn.click(fn=search_by_text, inputs=textbox, outputs=gallery)

    with gr.Tab("Buscar por imagen"):
        image_input = gr.Image(label="Sube una imagen de referencia", type="pil")
        image_btn = gr.Button(value="Buscar similares")
        gallery2 = gr.Gallery(label="Resultados", columns=5)
        image_btn.click(fn=search_by_image, inputs=image_input, outputs=gallery2)


if __name__ == "__main__":
    demo.launch()
