# Base image - CUDA bilan
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies o'rnatish
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory yaratish
WORKDIR /app

# Python dependencies o'rnatish
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Application fayllarini ko'chirish
COPY handler.py .
COPY start.sh .

# start.sh ga execute permission berish
RUN chmod +x start.sh

# Modelni oldindan yuklash (optional, build vaqtida)
# Bu build vaqtini oshiradi lekin run vaqtini kamaytiradi
# RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True); \
#     AutoModelForCausalLM.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True)"

# Port expose qilish (optional)
EXPOSE 8000

# Container ishga tushganda start.sh ni bajarish
CMD ["/app/start.sh"]
