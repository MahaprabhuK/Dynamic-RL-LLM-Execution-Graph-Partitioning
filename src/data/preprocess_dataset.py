from datasets import load_from_disk
import pandas as pd
import os

# Load the downloaded dataset
dataset = load_from_disk("data/raw/alpaca_dataset")

# Convert to pandas DataFrame
df = dataset["train"].to_pandas()

# Create a single prompt column
df["prompt"] = df.apply(
    lambda row: (
        row["instruction"]
        if row["input"] == ""
        else f"{row['instruction']}\n\n{row['input']}"
    ),
    axis=1
)

# Keep only required columns
processed_df = df[["prompt"]].copy()

# Add prompt IDs
processed_df.insert(0, "prompt_id", range(1, len(processed_df) + 1))

# Create output directory
os.makedirs("data/processed", exist_ok=True)

# Save CSV
output_path = "data/processed/prompts.csv"
processed_df.to_csv(output_path, index=False)

print("\nPreprocessing completed successfully!")
print(f"Saved to: {output_path}")
print(f"Total prompts: {len(processed_df)}")

print("\nFirst 5 prompts:")
print(processed_df.head())