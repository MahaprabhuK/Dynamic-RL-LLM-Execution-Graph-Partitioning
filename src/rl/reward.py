def calculate_reward(latency: float, cpu_percent: float, ram_percent: float, max_latency_ref: float = 40.0):
    """
    Calculate empirical reward for RL scheduler based on measured runtime outcomes.

    State-free reward formula:
        reward = -(0.6 * (latency / max_latency_ref) + 0.2 * (cpu_percent / 100.0) + 0.2 * (ram_percent / 100.0))

    Args:
        latency: Measured end-to-end latency in seconds.
        cpu_percent: Measured average CPU percentage during execution.
        ram_percent: Measured RAM percentage during execution.
        max_latency_ref: Reference maximum latency scaling bound (default: 40.0s).

    Returns:
        float: Scalar reward value in range [-1.0, 0.0].
    """

    latency_penalty = min(latency / max_latency_ref, 1.5)
    cpu_penalty = min(cpu_percent / 100.0, 1.0)
    ram_penalty = min(ram_percent / 100.0, 1.0)

    reward = -(
        0.6 * latency_penalty +
        0.2 * cpu_penalty +
        0.2 * ram_penalty
    )

    return float(reward)


if __name__ == "__main__":

    reward = calculate_reward(
        latency=20.0,
        cpu_percent=70.0,
        ram_percent=95.0
    )

    print("Test empirical reward:", reward)