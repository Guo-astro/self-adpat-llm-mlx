from datasets import Dataset, DatasetDict
import random

# Total number of examples to generate
num_examples = 500

# Generate synthetic examples.
# For this example, we create a simple arithmetic question.
data = []
for i in range(num_examples):
    # Create a simple arithmetic prompt with random numbers
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    prompt = f"What is {a} + {b}?"
    answer = str(a + b)
    data.append({"prompt": prompt, "answer": answer})

# Split the data into 400 train, 50 validation, and 50 test examples
train_data = data[:400]
valid_data = data[400:450]
test_data = data[450:]

# Create datasets from the lists
train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)
test_dataset = Dataset.from_list(test_data)

# Combine the splits into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset,
    "test": test_dataset
})

# (Optional) Print the dataset structure
print(dataset)

# Push the dataset to the Hugging Face Hub.
# Replace "username/my-grpo-dataset" with your desired repository name.
repo_id = "Goastro/mlx-grpo-dataset"  # e.g., "yourusername/my-grpo-dataset"
dataset.push_to_hub(repo_id)
