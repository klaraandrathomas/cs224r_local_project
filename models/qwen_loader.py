# models/qwen_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_qwen(fp16=True, device_map="auto"):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map
    )
    return model
