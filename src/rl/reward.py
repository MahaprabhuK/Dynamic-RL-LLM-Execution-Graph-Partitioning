def calculate_reward(latency, cpu_percent, ram_percent):
    """
    Calculate the reward for the RL scheduler.

    Lower latency and lower resource utilization
    should produce a better reward.
    """

    latency_penalty = latency / 30.0
    cpu_penalty = cpu_percent / 100.0
    ram_penalty = ram_percent / 100.0

    reward = -(
        0.6 * latency_penalty +
        0.2 * cpu_penalty +
        0.2 * ram_penalty
    )

    return reward


if __name__ == "__main__":

    reward = calculate_reward(
        latency=20.0,
        cpu_percent=70.0,
        ram_percent=95.0
    )

    print("Test reward:", reward)