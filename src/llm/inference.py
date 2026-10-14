import os
import sys
import time
from collections import deque
from functools import lru_cache

_torch_lib = r"C:\Users\Pavithran\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
if os.path.exists(_torch_lib):
    try:
        os.add_dll_directory(_torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from transformers import DynamicCache
from transformers.masking_utils import create_causal_mask

from src.llm.load_model import load_model, load_tokenizer
from src.llm.partition import PartitionStrategy, apply_partition_strategy
from src.data.load_prompts import load_prompts
from src.telemetry.monitor import SegmentPagingTracker, measure_activation_handoff


@lru_cache(maxsize=1)
def get_model():
    """Load the tokenizer and model once, on first use."""
    return load_model()


def get_tokenizer():
    """Tokenizer only; does not load the model weights."""
    return load_tokenizer()


def _format_prompt(tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_response(prompt, max_new_tokens: int = 100, return_token_ids: bool = False):
    """Monolithic baseline generation using model.generate()."""
    tokenizer, model = get_model()

    inputs = tokenizer(
        _format_prompt(tokenizer, prompt),
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if return_token_ids:
        return response, generated_tokens.tolist()

    return response


def generate_partitioned_response(
    prompt: str,
    strategy=0,
    max_new_tokens: int = 100,
    do_sample: bool = False,
    return_token_ids: bool = False,
    scheduler=None,
    decision_interval: int = 10,
    decision_log: list | None = None,
    step_callback=None,
    ignore_eos: bool = False,
    emergency_latency_factor: float | None = None,
    conservative_strategy: int | str = 1,
    segment_log: list | None = None,
    segment_tokens: int = 10
):
    """
    Executes autoregressive generation using explicit single-node layer-wise partitioning.

    Executes actual layer slices corresponding to the specified PartitionStrategy,
    with each stage running under its own intra-op thread budget, and measures
    activation handoffs at the boundaries between stages.

    Intra-generation re-scheduling: if `scheduler` is given, it is asked for a
    strategy before the first token and again every `decision_interval` tokens
    (scheduler.decide(step=..., prompt=..., input_tokens=...) -> strategy).
    All strategies share the same weights and KV cache, so switching mid-request
    costs nothing and does not change the generated tokens.

    Args:
        prompt: User prompt string.
        strategy: Strategy ID (0, 1, 2) or PartitionStrategy instance or name.
            Ignored when a scheduler is given.
        max_new_tokens: Maximum tokens to generate (default: 100).
        do_sample: Deterministic (False) or sampling.
        return_token_ids: Also return the generated token IDs.
        scheduler: Optional policy object that re-selects the strategy during generation.
        decision_interval: Tokens between scheduler decisions.
        decision_log: Optional list that receives one dict per scheduler decision.
        step_callback: Optional function called with the step index before each token.
        ignore_eos: Keep generating until max_new_tokens (fixed-length benchmarks).
        emergency_latency_factor: Engine latency switch, off when None (the default).
            If a token takes longer than this factor times the median of the last
            10 tokens, the next token runs on `conservative_strategy`. Only allowed
            with an adaptive scheduler; raises ValueError for fixed strategies.
            Not to be confused with the scheduler's CPU emergency (emergency_cpu),
            which reacts to external CPU % at decision points.
        conservative_strategy: Arm used by the latency switch.
        segment_log: Optional list that receives one paging/RAM record per
            `segment_tokens` generated tokens (swap in/out, page faults, available MB).

    Returns:
        tuple: (decoded_response_string, stage_telemetry_records_list[, generated_token_ids])
    """
    if scheduler is not None and decision_interval <= 0:
        raise ValueError("decision_interval must be positive when a scheduler is used")
    if emergency_latency_factor is not None:
        if emergency_latency_factor <= 0:
            raise ValueError("emergency_latency_factor must be positive (None = off)")
        if scheduler is None or getattr(scheduler, "is_fixed", False):
            raise ValueError("The latency emergency switch is only allowed with an adaptive scheduler, "
                             "not with a fixed strategy")
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")

    tokenizer, model = get_model()

    def resolve(s):
        return s if isinstance(s, PartitionStrategy) else apply_partition_strategy(s)

    strat_obj = resolve(strategy)
    layer_ranges = strat_obj.get_layer_ranges()
    stage_threads = strat_obj.get_thread_counts()
    num_stages = len(layer_ranges)

    inputs = tokenizer(_format_prompt(tokenizer, prompt), return_tensors="pt")
    input_ids = inputs["input_ids"]
    # Raw prompt token count: the input_tokens feature used in training
    prompt_tokens = len(tokenizer.encode(prompt)) if scheduler is not None else None

    stage_metrics = []
    generated_tokens = []
    original_threads = torch.get_num_threads()

    try:
        with torch.no_grad():
            curr_input_ids = input_ids.clone()
            past_key_values = DynamicCache(config=model.config)
            recent_token_latencies = deque(maxlen=10)
            paging = SegmentPagingTracker() if segment_log is not None else None
            segment_start, t_segment = 0, time.perf_counter()
            # Emergency kind recorded for the token that runs right after a trigger
            pending_emergency = None

            for step in range(max_new_tokens):
                t_token_step_start = time.perf_counter()
                if step_callback is not None:
                    step_callback(step)

                token_emergency = pending_emergency
                pending_emergency = None
                if scheduler is not None and step % decision_interval == 0:
                    t_decide = time.perf_counter()
                    strat_obj = resolve(scheduler.decide(step=step, prompt=prompt, input_tokens=prompt_tokens))
                    layer_ranges = strat_obj.get_layer_ranges()
                    stage_threads = strat_obj.get_thread_counts()
                    num_stages = len(layer_ranges)
                    telemetry = dict(getattr(scheduler, "last_telemetry", {}))
                    cpu_emergency = bool(telemetry.pop("emergency_switch", False))
                    if cpu_emergency:
                        token_emergency = "cpu"
                    if decision_log is not None:
                        decision_log.append({
                            "step": step,
                            "strategy_id": strat_obj.strategy_id,
                            "strategy": strat_obj.arm_name,
                            "thread_counts": stage_threads,
                            "thread_budget": strat_obj.thread_budget,
                            "decision_overhead_seconds": round(time.perf_counter() - t_decide, 6),
                            **telemetry,
                            "emergency_trigger": cpu_emergency,
                            "emergency_kind": "cpu" if cpu_emergency else ""
                        })

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

                # Execute layer partitions sequentially, each with its own thread budget
                for p_idx, ((start_layer, end_layer), num_threads) in enumerate(zip(layer_ranges, stage_threads), start=1):
                    thread_config_start = time.perf_counter()
                    if torch.get_num_threads() != num_threads:
                        torch.set_num_threads(num_threads)
                    thread_config_overhead = time.perf_counter() - thread_config_start

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

                    # Only stages followed by another stage have a real handoff boundary
                    is_boundary = p_idx < num_stages
                    handoff_record = measure_activation_handoff(
                        tensor=hidden_states if is_boundary else None,
                        stage_id=p_idx,
                        layer_start=start_layer,
                        layer_end=end_layer,
                        stage_latency_seconds=stage_duration,
                        num_threads=num_threads
                    )
                    handoff_record["token_step"] = step
                    handoff_record["strategy_id"] = strat_obj.strategy_id
                    handoff_record["arm"] = strat_obj.arm_name
                    handoff_record["emergency_trigger"] = token_emergency is not None
                    handoff_record["emergency_kind"] = token_emergency or ""
                    handoff_record["thread_config_overhead_seconds"] = round(thread_config_overhead, 8)
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

                step_duration = time.perf_counter() - t_token_step_start
                if hasattr(scheduler, "on_token"):
                    scheduler.on_token(step, step_duration)

                # Latency emergency switch (engine): if this token took more than
                # factor x the median of the last 10, the next token runs on the
                # conservative arm, bypassing the decision interval and hysteresis.
                if emergency_latency_factor is not None and len(recent_token_latencies) >= 5:
                    med_latency = float(np.median(recent_token_latencies))
                    conservative = resolve(conservative_strategy)
                    already_conservative = (conservative.strategy_id, tuple(conservative.get_thread_counts())) == \
                        (strat_obj.strategy_id, tuple(stage_threads))
                    if step_duration > emergency_latency_factor * med_latency and not already_conservative:
                        strat_obj = conservative
                        layer_ranges = strat_obj.get_layer_ranges()
                        stage_threads = strat_obj.get_thread_counts()
                        num_stages = len(layer_ranges)
                        pending_emergency = "latency"
                        if hasattr(scheduler, "on_emergency_switch"):
                            scheduler.on_emergency_switch(conservative_strategy)
                        if decision_log is not None:
                            decision_log.append({
                                "step": step + 1,
                                "strategy_id": strat_obj.strategy_id,
                                "strategy": strat_obj.arm_name,
                                "thread_counts": stage_threads,
                                "thread_budget": strat_obj.thread_budget,
                                "decision_overhead_seconds": 0.0,
                                "token_duration": round(step_duration, 4),
                                "recent_median": round(med_latency, 4),
                                "emergency_trigger": True,
                                "emergency_kind": "latency"
                            })

                recent_token_latencies.append(step_duration)

                if paging is not None and (step + 1 - segment_start >= segment_tokens or step == max_new_tokens - 1):
                    now = time.perf_counter()
                    segment_log.append({"segment": len(segment_log), "start_step": segment_start, "end_step": step,
                                        "segment_seconds": round(now - t_segment, 6), **paging.sample()})
                    segment_start, t_segment = step + 1, now

                if next_token_item == tokenizer.eos_token_id and not ignore_eos:
                    if paging is not None and segment_start <= step:
                        now = time.perf_counter()
                        segment_log.append({"segment": len(segment_log), "start_step": segment_start, "end_step": step,
                                            "segment_seconds": round(now - t_segment, 6), **paging.sample()})
                    break
    finally:
        torch.set_num_threads(original_threads)

    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if return_token_ids:
        return response_text, stage_metrics, generated_tokens

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
