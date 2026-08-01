import pandas as pd


def load_prompts(csv_path="data/processed/prompts.csv", limit=None):
    """
    Load prompts from the processed CSV.

    Args:
        csv_path (str): Path to prompts.csv
        limit (int): Number of prompts to load

    Returns:
        list[str]
    """

    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)

    return df["prompt"].tolist()


if __name__ == "__main__":

    prompts = load_prompts(limit=5)

    print(f"Loaded {len(prompts)} prompts.\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}\n")