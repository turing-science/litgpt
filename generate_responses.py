import json
import torch
from pathlib import Path
from litgpt import GPT, Config, Tokenizer
from litgpt.generate.base import generate
from litgpt.scripts.merge_lora import merge_lora
from litgpt.utils import load_checkpoint


def generate_responses(checkpoint_dir: Path, instructions: list, output_file: str, max_new_tokens: int = 100):
    """
    Generate responses for a list of instructions using the fine-tuned model and save to JSON.

    Args:
        checkpoint_dir: Path to the merged checkpoint directory
        instructions: List of instruction strings
        output_file: Path to save the JSON output
        max_new_tokens: Maximum number of new tokens to generate
    """
    # Load config and tokenizer
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    tokenizer = Tokenizer(checkpoint_dir)

    # Load model
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
        model = GPT(config)
        load_checkpoint(model, checkpoint_dir / "lit_model.pth")

    model.eval()

    results = []
    for instruction in instructions:
        # Prepare prompt (assuming Alpaca format, adjust if needed)
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

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

        # Clean up the response (remove extra newlines, etc.)
        response = generated_text.strip()

        # Create result entry
        result = {
            "dataset": "helpful_base",  # You can change this as needed
            "instruction": instruction,
            "output": response,
            "generator": "SmolLM2-135M-Instruct-finetuned"  # Adjust model name as needed
        }
        results.append(result)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Generated responses saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    # First, you need to merge the LoRA checkpoint
    # Assuming the finetuned checkpoint is in out/finetune/lora/final

    checkpoint_dir = Path("out/finetune/lora/final")  # Adjust path as needed

    # Example instructions (replace with your own)
    instructions = [
        "What are the names of some famous actors that started their careers on Broadway?",
        "How did US states get their names?"
    ]

    output_file = "generated_responses.json"

    generate_responses(checkpoint_dir, instructions, output_file)