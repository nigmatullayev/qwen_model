import runpod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import base64
import io
import os

# Global variables for model
model = None
tokenizer = None
device = None


def load_model():
    """Model va tokenizerni yuklash"""
    global model, tokenizer, device

    print("Loading Qwen-Image model...")

    # Device aniqlash
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "Qwen/Qwen-VL-Chat"

    # Tokenizer yuklash
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Model yuklash
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).eval()

    print("Model successfully loaded!")
    return model, tokenizer


def decode_base64_image(base64_string):
    """Base64 stringni PIL Image ga o'zgartirish"""
    try:
        # Base64 prefiksini olib tashlash (agar mavjud bo'lsa)
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")


def handler(job):
    """
    RunPod handler function

    Expected input format:
    {
        "input": {
            "prompt": "Describe this image",
            "image": "base64_encoded_image_string",
            "max_new_tokens": 512,
            "temperature": 0.7
        }
    }
    """
    global model, tokenizer, device

    try:
        # Model yuklash (agar yuklanmagan bo'lsa)
        if model is None or tokenizer is None:
            load_model()

        # Input ma'lumotlarini olish
        job_input = job["input"]
        prompt = job_input.get("prompt", "Describe this image in detail.")
        image_base64 = job_input.get("image")
        max_new_tokens = job_input.get("max_new_tokens", 512)
        temperature = job_input.get("temperature", 0.7)

        if not image_base64:
            return {"error": "No image provided"}

        # Rasmni dekodlash
        image = decode_base64_image(image_base64)

        # Rasmni vaqtinchalik saqlash
        temp_image_path = "/tmp/temp_image.jpg"
        image.save(temp_image_path)

        # Query tayyorlash
        query = tokenizer.from_list_format([
            {'image': temp_image_path},
            {'text': prompt},
        ])

        # Inference
        with torch.no_grad():
            response, history = model.chat(
                tokenizer,
                query=query,
                history=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

        # Vaqtinchalik faylni o'chirish
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return {
            "response": response,
            "prompt": prompt,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


# RunPod serverless handler
if __name__ == "__main__":
    # Model yuklash
    load_model()
    # RunPod serverless ishga tushirish
    runpod.serverless.start({"handler": handler})