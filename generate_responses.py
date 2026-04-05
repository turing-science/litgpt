import json
import torch
from pathlib import Path
import requests
from litgpt import GPT, Config, Tokenizer
from litgpt.generate.base import generate
from litgpt.utils import auto_download_checkpoint
from litgpt.prompts import PromptStyle


def load_alpaca_data(num_samples: int = 10):
    """Load the first num_samples from Alpaca dataset."""
    url = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json"
    download_dir = Path("./data/alpaca")
    download_dir.mkdir(parents=True, exist_ok=True)
    file_path = download_dir / "alpaca_data_cleaned_archive.json"

    if not file_path.exists():
        print(f"Downloading Alpaca data to {file_path}")
        response = requests.get(url)
        response.raise_for_status()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data[:num_samples]


def generate_responses(checkpoint_dir: str, output_file: str, max_new_tokens: int = 100, num_samples: int = 10):
    """
    Generate responses for Alpaca dataset samples using the model and save to JSON.

    Args:
        checkpoint_dir: HuggingFace model name or path to checkpoint directory
        output_file: Path to save the JSON output
        max_new_tokens: Maximum number of new tokens to generate
        num_samples: Number of samples to generate from Alpaca dataset
    """
    # Download/load checkpoint
    checkpoint_path = auto_download_checkpoint(checkpoint_dir)

    # Load config and tokenizer
    config = Config.from_file(checkpoint_path / "model_config.yaml")
    tokenizer = Tokenizer(checkpoint_path)

    # Load model
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model = GPT(config)
        from litgpt.utils import load_checkpoint
        load_checkpoint(model, checkpoint_path / "lit_model.pth")

    model.eval()

    # Load Alpaca data
    alpaca_samples = load_alpaca_data(num_samples)

    # Prompt style
    prompt_style = PromptStyle.from_name("alpaca")

    results = []
    for sample in alpaca_samples:
        instruction = sample["instruction"]
        input_text = sample.get("input", "")

        # Apply prompt style
        prompt = prompt_style.apply(instruction, input_text)

        # Encode
        encoded = tokenizer.encode(prompt, device=model.device)

        # Generate
        with torch.no_grad():
            output = generate(
                model, encoded,
                max_returned_tokens=len(encoded) + max_new_tokens,
                temperature=0.8,
                eos_id=tokenizer.eos_id
            )

        # Decode the generated part
        generated_text = tokenizer.decode(output[len(encoded):])

        # Clean up the response
        response = generated_text.strip()

        # Create result entry
        result = {
            "dataset": "helpful_base",  # You can change this as needed
            "instruction": instruction,
            "output": response,
            "generator": checkpoint_dir  # Use the model name
        }
        results.append(result)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Generated responses saved to {output_file}")


if __name__ == "__main__":
    # Example usage for base model
    checkpoint_dir = "HuggingFaceTB/SmolLM2-135M-Instruct"
    output_file = "base_model_responses.json"
    num_samples = 10  # Adjust as needed

    generate_responses(checkpoint_dir, output_file, num_samples=num_samples)