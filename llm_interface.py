import ollama
import json

class LLMInterface:
    def __init__(self, model_name='medictron-7b'):
        self.model_name = model_name
        print(f"[LLM] Initialized with Ollama model: {self.model_name}")

    def generate_structured_json(self, prompt, schema):
        """
        Uses the Ollama service to generate text and attempts to extract JSON.
        """
        system_prompt = (
            "You are a clinical information extraction specialist. "
            "Your task is to extract specific clinical features from the provided text. "
            f"You must respond ONLY with a valid JSON object that follows this schema: {json.dumps(schema)}. "
            "Do not include any conversational text or markdown formatting."
        )

        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nJSON Output:"

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                format='json'
            )
            
            content = response['response']
            
            # Extract JSON from the response (in case the model includes the prompt)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                print(f"[LLM Error] No JSON found in response: {content}")
                return None
        except Exception as e:
            print(f"[LLM Error] Failed to generate JSON via Ollama: {e}")
            return None

if __name__ == "__main__":
    # Quick test
    interface = LLMInterface()
    test_prompt = "Patient age 25, symptoms: fever and headache."
    test_schema = {"age": "int", "symptoms": "list"}
    
    print("Testing Ollama LLM Interface...")
    result = interface.generate_structured_json(test_prompt, test_schema)
    print(f"Result: {result}")
