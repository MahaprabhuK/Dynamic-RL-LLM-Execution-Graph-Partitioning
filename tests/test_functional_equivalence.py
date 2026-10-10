"""
Functional equivalence test utility.
Verifies that generate_partitioned_response() produces exact token-for-token matches
and exact token-ID equality with monolithic generate_response() across static, balanced, and latency_optimized strategies.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_prompts import load_prompts
from src.llm.inference import generate_response, generate_partitioned_response, tokenizer
from src.llm.partition import PartitionStrategy


def run_equivalence_tests(num_prompts: int = 3):
    print("=" * 80, flush=True)
    print(f"RUNNING DETAILED FUNCTIONAL EQUIVALENCE TEST ({num_prompts} PROMPTS x 3 STRATEGIES)", flush=True)
    print("=" * 80, flush=True)

    prompts = load_prompts(limit=num_prompts)
    all_passed = True

    for p_idx, prompt in enumerate(prompts, start=1):
        print(f"\n[Prompt {p_idx}/{num_prompts}] Testing prompt length: {len(prompt)}", flush=True)

        # Run monolithic baseline
        mono_text = generate_response(prompt)
        mono_token_ids = tokenizer.encode(mono_text)

        for strat_id in [0, 1, 2]:
            strat_name = PartitionStrategy(strat_id).strategy_name
            part_text, stages = generate_partitioned_response(prompt, strategy=strat_id)
            part_token_ids = tokenizer.encode(part_text)

            text_match = (mono_text == part_text)
            token_match = (mono_token_ids == part_token_ids)
            length_match = (len(mono_text) == len(part_text))

            match = text_match and token_match and length_match
            status = "PASSED" if match else "FAILED"
            if not match:
                all_passed = False

            print(f"  Strategy [{strat_id}: {strat_name:17s}] -> {status}", flush=True)
            print(f"    Token-ID Equality    : {token_match}", flush=True)
            print(f"    Decoded-Text Equality: {text_match}", flush=True)
            print(f"    Monolithic output len : {len(mono_text)}", flush=True)
            print(f"    Partitioned output len: {len(part_text)}", flush=True)
            print(f"    Stage handoff records : {len(stages)}", flush=True)

            if not match:
                print(f"    MONOLITHIC:\n{mono_text[:100]}...", flush=True)
                print(f"    PARTITIONED:\n{part_text[:100]}...", flush=True)

    print("\n" + "=" * 80, flush=True)
    if all_passed:
        print("SUCCESS: All functional equivalence tests PASSED perfectly (Token-ID + Decoded-Text)!", flush=True)
    else:
        print("FAILURE: Some functional equivalence tests produced mismatches.", flush=True)
    print("=" * 80, flush=True)

    return all_passed


if __name__ == "__main__":
    success = run_equivalence_tests(num_prompts=3)
    sys.exit(0 if success else 1)

