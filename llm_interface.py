import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import json

class LLMInterface:
    def __init__(self, model_name='nikitaredy/medictron-7B'):
        print(f"[LLM] Loading model: {model_name} (this may take a while...)")
        try:
            # Attempt 1: Standard AutoTokenizer (might fail due to TokenizersBackend error)
            print("[LLM] Attempting AutoTokenizer loading...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        except Exception as e:
            print(f"[LLM Warning] AutoTokenizer failed: {e}")
            print("[LLM] Attempting LlamaTokenizer without extra arguments to bypass error...")
            try:
                # Attempt 2: LlamaTokenizer without passing special_tokens to avoid the library's internal error
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e2:
                print(f"[LLM Error] All loading methods failed: {annotated_error(e2)}")
                raise e2

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.auto, 
            device_map="auto",
            trust_remote_code=True
        )
        print("[LLM] Model loaded successfully.")

def annotated_error(e):
    return str(e)

    def generate_structured_json(self, prompt, schema):
        """
        Uses the Transformers pipeline to generate text and attempts to extract JSON.
        """
        system_prompt = (
            "You are a clinical information extraction specialist. "
            "Your task is to extract specific clinical features from the provided text. "
            f"You must respond ONLY with a valid JSON object that follows this schema: {json.dumps(schema)}. "
            "Do not include any conversational text or markdown formatting."
        )

        full_prompt = f"{system_prompt}\n\nText to analyze: {prompt}\n\nJSON Output:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=500, 
                temperature=0.1, 
                do_sample=False
            )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from the response (in case the model includes the prompt)
        try:
            # Find the start of the JSON object
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                print(f"[LLM Error] No JSON found in response: {response_text}")
                return None
        except Exception as e:
            print(f"[LLM Error] Failed to parse JSON: {e}")
            return None

if __name__ == "__main__":
    # Quick test
    interface = LLMInterface()
    test_prompt = "Patient age 25, symptoms: fever and headache."
    test_schema = {"age": "int", "symptoms": "list"}
    
    print("Testing Transformers LLM Interface...")
    result = interface.generate_structured_json(test_prompt, test_schema)
    print(f"Result: {result}")
