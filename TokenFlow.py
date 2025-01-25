import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser(description="TokenFlow: Stream text generation with local models.")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration JSON file.")
args = parser.parse_args()

# Load configuration
with open(args.config, "r") as config_file:
    config = json.load(config_file)

model_name = config["model_name"]
max_tokens = config.get("max_tokens", 50)  # Default to 50 if not set

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("TokenFlow is running. Type your prompt below (type 'exit' to quit):")

# Main loop for continuous interaction
while True:
    input_prompt = input("PROMPT: ")
    if input_prompt.lower() == "exit":
        print("Goodbye!")
        break

    # Tokenize input
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

    # Token-by-token generation
    output_ids = input_ids.clone()
    print("Model:", end=" ", flush=True)
    for _ in range(max_tokens):
        outputs = model.generate(output_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        next_token = outputs[0, -1].unsqueeze(0).unsqueeze(0)
        output_ids = torch.cat([output_ids, next_token], dim=-1)
        print(tokenizer.decode(next_token[0]), end="", flush=True)
    print()  # For spacing between responses
