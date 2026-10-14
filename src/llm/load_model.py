from functools import lru_cache

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.config import load_config

MODEL_NAME = load_config()["model"]["name"]


@lru_cache(maxsize=1)
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model():
    model_config = load_config()["model"]
    dtype = getattr(torch, model_config["dtype"])

    print(f"Loading {MODEL_NAME} ({model_config['dtype']})...")

    tokenizer = load_tokenizer()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype
    )

    model.eval()

    print("Model loaded successfully!")

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_model()
