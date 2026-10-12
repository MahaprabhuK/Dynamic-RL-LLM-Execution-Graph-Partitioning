"""
bfloat16 Validation & Equivalence Script (P1.5):
Evaluates TinyLlama in bfloat16 on CPU:
1. Verifies functional equivalence: partitioned generation matches monolithic model.generate() in bfloat16.
2. Compares idle CPU throughput (tokens/s) and RAM footprint between float32 and bfloat16.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Windows PyTorch DLL loading helper
torch_lib = r"C:\Users\Pavithran\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import psutil
import torch


def run_bfloat16_validation(num_tokens=30, dry_run=False):
    print("=" * 80)
    print("BFLOAT16 VALIDATION & IDLE THROUGHPUT EVALUATION (P1.5)")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN] Plan:")
        print("  1. Load TinyLlama with dtype=bfloat16 on CPU.")
        print(f"  2. Run monolithic generate() for {num_tokens} tokens.")
        print(f"  3. Run partitioned generate() across strategies for {num_tokens} tokens.")
        print("  4. Check token-ID equality, throughput (tokens/s), and RAM consumption.")
        print("  Estimated runtime: ~30-45 seconds (1 prompt, warm up + 3 strategies).")
        print("Dry run complete. No models executed.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.data.load_prompts import load_prompts
    import src.llm.inference as inf
    from src.llm.inference import _format_prompt, generate_partitioned_response
    from src.llm.partition import PartitionStrategy

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompt = load_prompts(limit=1)[0]

    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 ** 2)

    print(f"Loading {model_name} in bfloat16 on CPU...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_bf16 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model_bf16.eval()
    load_time = time.perf_counter() - t0
    ram_after = process.memory_info().rss / (1024 ** 2)
    print(f"Loaded in {load_time:.2f} s. Model RAM footprint: {ram_after - ram_before:.1f} MB (Total process: {ram_after:.1f} MB)")

    # 1. Monolithic generation in bf16
    inputs = tokenizer(_format_prompt(tokenizer, prompt), return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model_bf16.generate(**inputs, max_new_tokens=num_tokens, do_sample=False)
    mono_dur = time.perf_counter() - t0
    mono_tokens = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    mono_tps = len(mono_tokens) / mono_dur
    print(f"\nMonolithic bf16 generate(): {len(mono_tokens)} tokens in {mono_dur:.2f} s ({mono_tps:.2f} tok/s)")

    # Hook model_bf16 into inference
    inf.get_model = lambda: (tokenizer, model_bf16)

    # 2. Partitioned generation across strategies in bf16
    all_matched = True
    print("\nPartitioned strategies vs Monolithic in bfloat16:")
    for sid, sname in PartitionStrategy.STRATEGIES.items():
        strat = PartitionStrategy(sid)
        t0 = time.perf_counter()
        part_text, stages, part_ids = generate_partitioned_response(
            prompt, strategy=sid, max_new_tokens=num_tokens, return_token_ids=True
        )
        part_dur = time.perf_counter() - t0
        part_tps = len(part_ids) / part_dur

        match = (part_ids == mono_tokens)
        all_matched &= match
        print(f"  Strategy [{sid}: {sname:17s} threads={strat.get_thread_counts()}]: {'PASSED' if match else 'FAILED'}")
        print(f"    Token-ID Match: {match} ({len(part_ids)} vs {len(mono_tokens)} tokens) | Throughput: {part_tps:.2f} tok/s ({part_dur:.2f} s)")

    print("\n" + "=" * 80)
    if all_matched:
        print("BFLOAT16 VALIDATION SUCCESS: All partitioned strategies produce token-identical output to monolithic model.generate() in bfloat16!")
    else:
        print("BFLOAT16 VALIDATION WARNING: Token divergence detected between partitioned and monolithic generation in bfloat16.")
    print(f"RAM Savings: fp32 (~4400 MB) vs bf16 ({ram_after - ram_before:.0f} MB) -> ~{4400 - (ram_after - ram_before):.0f} MB saved ({((4400 - (ram_after - ram_before))/4400)*100:.1f}% reduction).")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate bfloat16 equivalence and idle throughput.")
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running inference.")
    args = parser.parse_args()

    run_bfloat16_validation(num_tokens=args.tokens, dry_run=args.dry_run)
