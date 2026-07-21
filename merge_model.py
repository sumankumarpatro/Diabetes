import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from loguru import logger
from pathlib import Path

def merge_adapter_to_base(base_model_base_name: str, adapter_path: str, output_path: str) -> None:
    """
    Merges a LoRA adapter into a base model and saves the result.
    """
    base_model_path = Path(base_model_base_name)
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    if not adapter_path.exists():
        logger.error(f"Error: Adapter path {adapter_path} does not exist.")
        return

    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading base model: {base_model_base_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_base_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_base_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path), trust_remote_code=True)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    logger.success("Merge complete!")

if __name__ == "__main__":
    # Configuration
    BASE_MODEL = "BioMistral/BioMistral-7B"
    ADAPTER_PATH = "./medictron-7B"
    OUTPUT_PATH = "./medic_merged"

    merge_adapter_to_base(BASE_MODEL, ADAPTER_PATH, OUTPUT_PATH)
