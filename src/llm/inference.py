from src.llm.load_model import load_model
from src.data.load_prompts import load_prompts
import torch

# Load model ONLY ONCE
tokenizer, model = load_model()


def generate_response(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


if __name__ == "__main__":

    prompts = load_prompts(limit=1)

    print("=" * 80)
    print("PROMPT\n")
    print(prompts[0])

    print("\n" + "=" * 80)

    response = generate_response(prompts[0])

    print("MODEL RESPONSE\n")
    print(response)