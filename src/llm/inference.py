import time
import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import create_causal_mask

from src.llm.load_model import load_model
from src.llm.partition import PartitionStrategy, apply_partition_strategy
from src.data.load_prompts import load_prompts
from src.telemetry.monitor import measure_activation_handoff

# Load model ONLY ONCE
tokenizer, model = load_model()


def generate_response(prompt):
    """Monolithic baseline generation using model.generate()."""
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


def generate_partitioned_response(prompt: str, strategy=0, max_new_tokens: int = 100, do_sample: bool = False):
    """
    Executes autoregressive generation using explicit single-node layer-wise partitioning.

    Executes actual layer slices corresponding to the specified PartitionStrategy,
    measuring physical activation handoffs at partition boundaries.

    Args:
        prompt: User prompt string.
        strategy: Strategy ID (0, 1, 2) or PartitionStrategy instance or name.
        max_new_tokens: Maximum tokens to generate (default: 100).
        do_sample: Deterministic (False) or sampling.

    Returns:
        tuple: (decoded_response_string, stage_telemetry_records_list)
    """
    if not isinstance(strategy, PartitionStrategy):
        strat_obj = apply_partition_strategy(strategy)
    else:
        strat_obj = strategy

    layer_ranges = strat_obj.get_layer_ranges()

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    stage_metrics = []
    generated_tokens = []

    with torch.no_grad():
        curr_input_ids = input_ids.clone()
        past_key_values = DynamicCache(config=model.config)

        for step in range(max_new_tokens):
            if step == 0:
                seq_ids = curr_input_ids
                position_ids = torch.arange(seq_ids.shape[1], device=seq_ids.device).unsqueeze(0)
            else:
                seq_ids = curr_input_ids[:, -1:]
                past_seen_tokens = past_key_values.get_seq_length()
                position_ids = torch.tensor([[past_seen_tokens]], device=seq_ids.device)

            inputs_embeds = model.model.embed_tokens(seq_ids)
            causal_mask = create_causal_mask(
                config=model.config,
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=past_key_values,
                position_ids=position_ids
            )
            position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids=position_ids)

            hidden_states = inputs_embeds

            # Execute layer partitions sequentially
            for p_idx, (start_layer, end_layer) in enumerate(layer_ranges, start=1):
                t_stage_start = time.perf_counter()

                for idx in range(start_layer, end_layer + 1):
                    decoder_layer = model.model.layers[idx]
                    layer_out = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                stage_duration = time.perf_counter() - t_stage_start

                # Perform and measure physical activation copy/handoff across partition boundary
                handoff_record = measure_activation_handoff(
                    tensor=hidden_states,
                    stage_id=p_idx,
                    layer_start=start_layer,
                    layer_end=end_layer,
                    stage_latency_seconds=stage_duration
                )
                stage_metrics.append(handoff_record)

            hidden_states = model.model.norm(hidden_states)
            logits = model.lm_head(hidden_states[:, -1, :])

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_token_item = next_token.item()
            generated_tokens.append(next_token_item)
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=-1)

            if next_token_item == tokenizer.eos_token_id:
                break

    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response_text, stage_metrics


if __name__ == "__main__":
    prompts = load_prompts(limit=1)

    print("=" * 80)
    print("PROMPT\n")
    print(prompts[0])

    print("\n" + "=" * 80)

    for strat_id in [0, 1, 2]:
        resp, stages = generate_partitioned_response(prompts[0], strategy=strat_id, max_new_tokens=10)
        print(f"\nStrategy [{strat_id}] Output:\n{resp}")
        print(f"Recorded {len(stages)} stage handoff records.")