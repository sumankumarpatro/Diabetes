import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_adapter_to_base(base_model_name, adapter_path, output_path):
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu", # Use CPU to avoid OOM on many machines during merge
        trust_remote_code=True
    )

    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, trust_remote_remote_code=True)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete!")

if __name__ == "__main__":
    # Configuration
    BASE_MODEL = "BioMistral/BioMistral-7B"
    ADAPTER_PATH = "./medictron-7B"
    OUTPUT_PATH = "./medictron-7B-merged"

    if not os.path.exists(ADAPTER_PATH):
        print(f"Error: Adapter path {ADAPTER_PATH} does not exist.")
    else:
        merge_adapter_to_base(BASE_MODEL, ADAPTER_PATH, OUTPUT_PATH)
