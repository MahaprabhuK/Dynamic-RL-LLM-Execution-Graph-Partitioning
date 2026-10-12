"""
Functional equivalence test.
Verifies that generate_partitioned_response() produces exactly the same
generated token IDs (and decoded text) as monolithic model.generate() for
every partition strategy.

Run with pytest (loads the model, takes a few minutes on CPU):
    python -m pytest tests/test_functional_equivalence.py -v
or as a script:
    python tests/test_functional_equivalence.py
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_prompts import load_prompts
from src.llm.partition import PartitionStrategy

NUM_PROMPTS = 3
MAX_NEW_TOKENS = 100


@pytest.fixture(scope="module")
def model_runtime():
    from src.llm.inference import generate_response, generate_partitioned_response

    return generate_response, generate_partitioned_response


@pytest.fixture(scope="module")
def monolithic_outputs(model_runtime):
    generate_response, _ = model_runtime
    prompts = load_prompts(limit=NUM_PROMPTS)
    return [
        (prompt, *generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS, return_token_ids=True))
        for prompt in prompts
    ]


@pytest.mark.parametrize("strategy_id", list(PartitionStrategy.STRATEGIES))
def test_partitioned_matches_monolithic(monolithic_outputs, model_runtime, strategy_id):
    _, generate_partitioned_response = model_runtime
    for prompt, mono_text, mono_ids in monolithic_outputs:
        part_text, stages, part_ids = generate_partitioned_response(
            prompt, strategy=strategy_id, max_new_tokens=MAX_NEW_TOKENS, return_token_ids=True
        )
        assert part_ids == mono_ids, f"Token IDs differ for prompt {prompt[:40]!r}"
        assert part_text == mono_text
        assert len(stages) > 0


def run_equivalence_tests(num_prompts: int = NUM_PROMPTS):
    from src.llm.inference import generate_response, generate_partitioned_response

    print("=" * 80, flush=True)
    print(f"RUNNING FUNCTIONAL EQUIVALENCE TEST ({num_prompts} PROMPTS x 3 STRATEGIES)", flush=True)
    print("=" * 80, flush=True)

    all_passed = True

    for p_idx, prompt in enumerate(load_prompts(limit=num_prompts), start=1):
        print(f"\n[Prompt {p_idx}/{num_prompts}] Testing prompt length: {len(prompt)}", flush=True)
        mono_text, mono_ids = generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS, return_token_ids=True)

        for strat_id in PartitionStrategy.STRATEGIES:
            strat = PartitionStrategy(strat_id)
            part_text, stages, part_ids = generate_partitioned_response(
                prompt, strategy=strat_id, max_new_tokens=MAX_NEW_TOKENS, return_token_ids=True
            )

            token_match = part_ids == mono_ids
            text_match = part_text == mono_text
            match = token_match and text_match
            all_passed &= match

            print(f"  Strategy [{strat_id}: {strat.strategy_name:17s} threads={strat.get_thread_counts()}] -> "
                  f"{'PASSED' if match else 'FAILED'}", flush=True)
            print(f"    Token-ID Equality    : {token_match} ({len(part_ids)} vs {len(mono_ids)} tokens)", flush=True)
            print(f"    Decoded-Text Equality: {text_match}", flush=True)
            print(f"    Stage records        : {len(stages)}", flush=True)

            if not match:
                first_diff = next((i for i, (a, b) in enumerate(zip(part_ids, mono_ids)) if a != b),
                                  min(len(part_ids), len(mono_ids)))
                print(f"    First differing token index: {first_diff}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUCCESS: all strategies match model.generate()" if all_passed
          else "FAILURE: some strategies produced different tokens", flush=True)
    print("=" * 80, flush=True)

    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_equivalence_tests() else 1)
