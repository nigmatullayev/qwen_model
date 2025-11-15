import torch
from diffusers import DiffusionPipeline
import runpod

# Model allaqachon Docker ichida bo‘ladi, shuning uchun localdan yuklanadi
pipe = DiffusionPipeline.from_pretrained(
    "/models/sdxl",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

def generate(job):
    prompt = job["input"]["prompt"]

    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=3.5
    ).images[0]

    file_path = f"/tmp/{job['id']}.png"
    image.save(file_path)

    return {"image_path": file_path}

runpod.serverless.start({"handler": generate})
