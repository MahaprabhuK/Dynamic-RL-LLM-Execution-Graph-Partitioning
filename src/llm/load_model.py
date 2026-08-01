from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model():
    print("Loading TinyLlama...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32
    )

    model.eval()

    print("Model loaded successfully!")

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_model()