import numpy as np

STATE_SIZE = 4


def build_state(prompt_length, input_tokens, cpu_percent, ram_percent):
    """
    Normalized pre-decision state shared by training, evaluation and the
    online scheduler, so the policy sees identical features everywhere:

        [prompt_length / 200, input_tokens / 100, cpu_percent / 100, ram_percent / 100]

    cpu_percent and ram_percent must be measured BEFORE the request runs.
    """

    return np.array([
        float(prompt_length) / 200.0,
        float(input_tokens) / 100.0,
        float(cpu_percent) / 100.0,
        float(ram_percent) / 100.0
    ], dtype=np.float32)


def observe_state(prompt, tokenizer, pre_metrics):
    """Build the state for a live request from a pre-run collect_metrics() snapshot."""

    return build_state(
        prompt_length=len(prompt),
        input_tokens=len(tokenizer.encode(prompt)),
        cpu_percent=pre_metrics["cpu_percent"],
        ram_percent=pre_metrics["ram_percent"]
    )
