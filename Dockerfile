FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Kerakli Python kutubxonalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modelni oldindan yuklab olish
RUN mkdir -p /models/sdxl
RUN python3 - <<EOF
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype="auto"
)
pipe.save_pretrained("/models/sdxl")
EOF

COPY handler.py .
COPY start.sh .
RUN chmod +x start.sh

CMD ["/workspace/start.sh"]
