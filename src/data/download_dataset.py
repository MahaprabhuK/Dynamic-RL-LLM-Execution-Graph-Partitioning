from datasets import load_dataset
import os

# Create data/raw directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

print("Downloading Stanford Alpaca dataset...")

# Download dataset
dataset = load_dataset("tatsu-lab/alpaca")

# Save locally
dataset.save_to_disk("data/raw/alpaca_dataset")

print("\nDataset downloaded successfully!")
print(dataset)