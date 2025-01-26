import argparse
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="TokenFlow: Stream text generation with local models.")
parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration JSON file.")
args = parser.parse_args()

# Load configuration
with open(args.config, "r") as config_file:
    config = json.load(config_file)

model_name = config["model_name"]
max_tokens = config.get("max_tokens", 50)  # Default to 50 if not set

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN + f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Move model to GPU 

print(Fore.GREEN + "TokenFlow is running. Type your prompt below (type 'exit' to quit):")


while True:
    input_prompt = input(Fore.BLUE + "PROMPT: " + Style.RESET_ALL)
    if input_prompt.lower() == "exit":
        print(Fore.RED + "Goodbye!")
        break

    # Tokenize inp and move to device
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)

    # Token-by-token generation with timing
    output_ids = input_ids.clone()
    start_time = time.time()
    print(Fore.YELLOW + "Model:", end=" ", flush=True)

    for _ in range(max_tokens):
        outputs = model.generate(output_ids, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
        next_token = outputs[0, -1].unsqueeze(0).unsqueeze(0)
        output_ids = torch.cat([output_ids, next_token], dim=-1)
        print(tokenizer.decode(next_token[0]), end="", flush=True)

    print()  #
    end_time = time.time()

    # Statistics
    total_tokens = output_ids.size(1) - input_ids.size(1)
    generation_time = end_time - start_time
    tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

    print(Fore.CYAN + "\n--- Statistics ---")
    print(Fore.MAGENTA + f"Total tokens generated: {total_tokens}")
    print(Fore.MAGENTA + f"Tokens per second: {tokens_per_second:.2f}")
    print(Fore.CYAN + "------------------" + Style.RESET_ALL)
