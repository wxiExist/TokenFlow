# TokenFlow

**TokenFlow** is a lightweight, interactive tool for working with local language models. It allows you to generate text token by token in real-time directly from the console. Perfect for developers and AI enthusiasts who want more control and flexibility with their language models.

---

## Features
- **Real-time token streaming**: Watch the text generate token by token for an interactive experience.
- **Local model support**: Use any Hugging Face-compatible model, from small GPT-2 to large GPT-Neo models.
- **Customizable configurations**: Configure model name and generation parameters through a JSON file.
- **Interactive conversation**: Keep the session alive and chat endlessly without restarting the script.

---

## Requirements
- Python 3.8 or higher
- PyTorch
- Hugging Face `transformers` library
- colorama 

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/wxiExist/TokenFlow.git
   cd TokenFlow
2. Install dependencies:
```bash
   pip install torch transformers
```
---
## Usage
1. Create a configuration file `config.json` in the project directory with the following format:
   ```json
   {
     "model_name": "gpt2",
     "max_tokens": 50
   }
