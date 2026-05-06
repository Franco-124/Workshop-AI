# setup.ps1
# Ejecutar desde la raíz del proyecto: .\setup.ps1

# 1. Crear el proyecto con uv
uv init clip_faiss_search
cd clip_faiss_search

# 2. Crear virtualenv con Python 3.11
uv venv --python 3.11

# 3. Activar el virtualenv
.venv\Scripts\Activate.ps1

# 4. Instalar dependencias
uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv add transformers
uv add faiss-cpu
uv add Pillow
uv add numpy
uv add requests
uv add gradio

# 5. Verificar instalación
python -c "import torch; import transformers; import faiss; import gradio; print('Todo instalado correctamente')"