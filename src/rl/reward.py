def calculate_reward(
    latency: float,
    cpu_percent: float = 0.0,
    ram_percent: float = 0.0,
    num_tokens: int | None = None,
    max_latency_ref: float = 40.0,
    sec_per_token_ref: float = 0.5,
    latency_weight: float = 1.0,
    cpu_weight: float = 0.0,
    ram_weight: float = 0.0
):
    """
    Calculate empirical reward for the RL scheduler based on measured runtime outcomes.

        reward = -(latency_weight * latency_penalty
                   + cpu_weight * cpu_percent / 100
                   + ram_weight * ram_percent / 100)

    If num_tokens is given, latency_penalty = (latency / num_tokens) / sec_per_token_ref,
    which makes rewards comparable across prompts with different response lengths.
    Otherwise latency_penalty = latency / max_latency_ref. Penalties are clipped.

    CPU/RAM weights default to 0: those values are single post-run snapshots and
    mostly add noise. Set them if resource usage should be traded off against latency.

    Returns:
        float: Scalar reward, <= 0 (higher is better).
    """

    if num_tokens:
        latency_penalty = min((latency / num_tokens) / sec_per_token_ref, 1.5)
    else:
        latency_penalty = min(latency / max_latency_ref, 1.5)
    cpu_penalty = min(cpu_percent / 100.0, 1.0)
    ram_penalty = min(ram_percent / 100.0, 1.0)

    reward = -(
        latency_weight * latency_penalty +
        cpu_weight * cpu_penalty +
        ram_weight * ram_penalty
    )

    return float(reward)


if __name__ == "__main__":

    print("Latency-only reward (20s):", calculate_reward(latency=20.0))
    print("Per-token reward (20s, 80 tokens):", calculate_reward(latency=20.0, num_tokens=80))
    print(
        "Weighted reward (v2 weights):",
        calculate_reward(20.0, 70.0, 95.0, latency_weight=0.6, cpu_weight=0.2, ram_weight=0.2)
    )
