import json
import re
import os
from rag_retriever import RAGRetriever
from llm_interface import LLMInterface

class ClinicalOrchestratorAgent:
    def __init__(self, index_path, docs_mapping_path, model_name):
        self.retriever = RAGRetriever(index_path, docs_mapping_path)
        self.retriever.load()
        self.llm = LLMInterface(model_name=model_name)
        self.extraction_schema = {
            "age_group": "string",
            "symptoms": "list of strings",
            "medications": "list of strings",
            "hospital_stay_days": "integer",
            "glucose_status": "string"
        }

    def orchestrate(self, clinical_note):
        print(f"\n[Orchestrator] Received Note: {clinical_note}")
        print("[Orchestrator] Retrieving medical context...")
        context_docs = self.retriever.retrieve(clinical_note, k=1)
        context_text = "\n".join(context_docs)
        print(f"[Retrieved Context]: {context_text}")
        augmented_text = f"Clinical Note: {clinical_note}\n\nMedical Context: {context_text}"
        print(f"[Augmented Text]: {augmented_text}")
        print("[Semantic Parser] Extracting features into JSON via LLM...")
        extracted_features = self._llm_parsing(augmented_text)
        
        return extracted_features

    def _llm_parsing(self, text):
        """
        Uses the LLMInterface to extract structured features from the augmented text.
        """
        prompt = f"Extract clinical features from the following text:\n\n{text}"
        return self.llm.generate_structured_json(prompt, self.extraction_schema)

    def _simulate_parsing(self, text):
        """
        Simulates an LLM parsing the text and extracting structured features.
        In reality, this would use an LLM prompt.
        """
        # Simple regex-based extraction to simulate the agent's capability
        features = {
            "age_group": "Unknown",
            "symptoms": [],
            "medications": [],
            "hospital_stay_days": None,
            "glucose_status": "Unknown"
        }

        # Extract age (e.exp: [0-10), [10-20), etc.)
        age_match = re.search(r'(\[\d+-\d+[\]\)]?|age \d+)', text)
        if age_match:
            features["age_group"] = age_match.group(0)

        # Extract symptoms (looking for keywords from our Hinglish notes)
        symptoms_keywords = ["sugar issues", "fever", "bukhar", "pain", "dard", "fatigue", "thakan"]
        for word in symptoms_keywords:
            if word.lower() in text.lower():
                features["symptoms"].append(word)

        # Extract hospital stay
        hosp_match = re.search(r'hospital stay: (\d+) days|Time in hospital: (\d+) days', text)
        if hosp_match:
            days = hosp_match.group(1) or hosp_match.group(2)
            features["hospital_stay_days"] = int(days)

        # Extract glucose status
        if "high sugar" in text.lower() or "hyperglycemia" in text.lower():
            features["glucose_status"] = "Hyperglycemia"
        elif "low sugar" in text.lower() or "hypoglycemia" in text.lower():
            features["glucose_status"] = "Hypoglycemia"

        return features

if __name__ == "__main__":
    # Configuration
    PROCESSED_DIR = "/Users/unasumankumarpatro/Documents/Diabetes/processed_data"
    INDEX_PATH = os.path.join(PROCESSED_DIR, 'medical_kb.index')
    DOCS_PATH = os.path.join(PROCESSED_DIR, 'medical_kb_docs.npy')
    MODEL_NAME = 'medictron-7b'

    # Initialize Agent
    agent = ClinicalOrchestratorAgent(INDEX_PATH, DOCS_PATH, model_name=MODEL_NAME)

    # Test with a sample Hinglish note (from our generated dataset)
    sample_note = "Patient age [0-10) presented with high sugar issues. Bukhar (fever) and weakness reported. Time in hospital: 4 days."
    
    print("\n--- Starting Clinical Decision Support Process ---")
    result_json = agent.orchestrate(sample_note)
    
    print("\n[Final Extracted JSON Payload]")
    print(json.dumps(result_json, indent=4))
