import argparse
import sys
from llm_providers import OllamaProvider
from loguru import logger
import json
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from rag_retriever import RAGRetriever
from llm_interface import LLMInterface
from config import config

class ClinicalFeatures(BaseModel):
    """
    Structured clinical features extracted from a clinical note.
    """
    age_group: str = Field(..., description="The age group of the patient (e.g., '[0-10)', 'age 25').")
    symptoms: List[str] = Field(default_factory=list, description="List of reported symptoms.")
    medications: List[str] = Field(default_factory=list, description="List of medications the patient is taking.")
    medication_count: Optional[int] = Field(None, description="The number of medications explicitly stated in the note.")
    condition_status: str = Field("Unknown", description="The patient's condition status as explicitly reported in the note.")
    hospital_stay_days: Optional[int] = Field(None, description="Number of days the patient stayed in the hospital.")
    glucose_status: str = Field("Unknown", description="The glucose status (e.g., 'Hyperglycemia', 'Hypoglycemia', 'Unknown').")

class ClinicalDecisionReport(BaseModel):
    """
    The final output of the Clinical Orchestrator, containing features, prediction, and recommendations.
    """
    features: ClinicalFeatures
    readmission_risk: Optional[bool]
    recommendations: str

class ClinicalOrchestratorAgent:
    def __init__(self, retriever: RAGRetriever, llm: LLMInterface):
        """
        Initializes the Orchestrator with injected dependencies.
        """
        self.retriever = retriever
        self.llm = llm
        # Reverted to simple types to avoid confusing the LLM with complex instructions in the schema
        self.extraction_schema = {
            "age_group": "The patient's age or age group (e.g., '30', '30-40', 'adult').",
            "symptoms": "A list of strings representing symptoms.",
            "medications": "A list of strings representing medications.",
            "medication_count": "An integer count of medications explicitly stated in the note, or null.",
            "condition_status": "A short phrase describing the patient's condition as stated in the note, or 'Unknown'.",
            "hospital_stay_days": "An integer or null.",
            "glucose_status": "The glucose status (e.g., 'Hyperglycemia', 'Hypoglycemia', 'Unknown')."
        }
        self.recommendation_schema = {
            "actionable_steps": "A list of strings, where each string is a clear, professional medical instruction (e.g., 'Monitor blood glucose levels daily').",
            "urgency_level": "A single word: 'Low', 'Medium', or 'High'.",
            "clinical_rationale": "A brief, professional explanation for the recommendations based on the clinical data."
        }

    def _extract_hospital_stay_days(self, clinical_note: str) -> Optional[int]:
        """
        Extracts hospital stay days from the note text when a clear numeric value is present.
        """
        import re

        patterns = [
            r'hospital\s+stay\s*[:=]\s*(\d+)',
            r'hospital\s+stay\s*(?:was|is|for)?\s*(\d+)',
            r'\b(\d+)\s+days?\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, clinical_note, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _extract_medication_count(self, clinical_note: str) -> Optional[int]:
        """
        Extracts the stated number of medications from the note.
        """
        import re

        patterns = [
            r'number of medications\s*[:=]\s*(\d+)',
            r'medications\s*[:=]\s*(\d+)',
            r'\b(\d+)\s+medications\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, clinical_note, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _extract_condition_status(self, clinical_note: str) -> Optional[str]:
        """
        Extracts explicit condition status phrases from the note.
        """
        import re

        status_patterns = [
            r'\b(stable|unstable|improving|worsening|deteriorating)\b',
        ]

        for pattern in status_patterns:
            match = re.search(pattern, clinical_note, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()

        return None

    def _merge_reflected_data(self, extracted_data: Optional[Dict[str, Any]], reflected_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preserves the initial extraction for already-populated fields and only uses reflected values for missing data.
        This avoids the reflection step overwriting values such as hospital_stay_days with hallucinated values.
        """
        if not isinstance(extracted_data, dict):
            return reflected_data or {}
        if not isinstance(reflected_data, dict):
            return extracted_data

        merged: Dict[str, Any] = {}
        for key in set(extracted_data.keys()) | set(reflected_data.keys()):
            original_value = extracted_data.get(key)
            reflected_value = reflected_data.get(key)

            is_missing = original_value is None or original_value == "" or (isinstance(original_value, list) and len(original_value) == 0)
            if not is_missing:
                merged[key] = original_value
            else:
                merged[key] = reflected_value

        return merged

    def orchestrate(self, clinical_note: str, readmission_prediction: Optional[bool] = None) -> Optional[ClinicalDecisionReport]:
        """
        Orchestrates the R/RAG and LLM pipeline to extract features, 
        incorporate prediction, and generate clinical recommendations.
        """
        print(f"DEBUG: Orchestrate called with note: {clinical_note[:50]}...")
        logger.info(f"Received Clinical Note: {clinical_note[:100]}...")
        
        try:
            # 1. RAG Retrieval
            logger.info("Retrieving medical context...")
            context_docs = self.retriever.retrieve(clinical_note, k=config.RETRIEVAL_K)
            context_text = "\n".join(context_docs)
            logger.debug(f"Retrieved Context: {context_text[:200]}...")

            # 2. Augmentation
            note_with_context = f"Clinical Note: {clinical_note}\n\nMedical Context: {context_text}"

            # 3. Semantic Parsing (Feature Extraction)
            logger.info("Extracting features via LLM from the note and retrieved medical context...")
            extracted_data = self._llm_parsing(note_with_context)
            
            if not extracted_data:
                logger.warning("Failed to extract features from note.")
                return None

            # Prefer the explicit hospital stay from the note over the LLM's raw extraction.
            note_hospital_stay_days = self._extract_hospital_stay_days(clinical_note)
            if note_hospital_stay_days is not None:
                extracted_data["hospital_stay_days"] = note_hospital_stay_days

            # --- NEW: REFLECTOR STEP (Self-Correction) ---
            logger.info("Reflecting on extraction accuracy...")
            validated_data = self._reflect_on_extraction(clinical_note, extracted_data)
            
            if not validated_data:
                logger.warning("Reflector failed to validate extraction. Falling pre-reflection data.")
                validated_data = extracted_data
            else:
                validated_data = self._merge_reflected_data(extracted_data, validated_data)
                logger.success("Reflector validated/corrected the extraction.")
            # ----------------------------------------------

            # 4. Final Report Construction
            # We now use the validated data to build the ClinicalDecisionReport
            # Normalize medication entries to a flat list of strings.
            medications_raw = validated_data.get("medications", [])
            normalized_medications = []
            if isinstance(medications_raw, list):
                for item in medications_raw:
                    if isinstance(item, list):
                        normalized_medications.extend(str(sub).strip() for sub in item if sub is not None)
                    elif item is not None:
                        normalized_medications.append(str(item).strip())
            elif medications_raw is not None:
                normalized_medications = [str(medications_raw).strip()]

            # Normalize symptoms to a list of strings when the LLM returns a comma-separated string.
            symptoms_raw = validated_data.get("symptoms", [])
            if isinstance(symptoms_raw, str):
                normalized_symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()]
            elif isinstance(symptoms_raw, list):
                normalized_symptoms = [str(item).strip() for item in symptoms_raw if item is not None]
            else:
                normalized_symptoms = []

            hospital_stay_days = validated_data.get("hospital_stay_days")
            if hospital_stay_days is None or hospital_stay_days == "":
                note_hospital_stay_days = self._extract_hospital_stay_days(clinical_note)
                if note_hospital_stay_days is not None:
                    hospital_stay_days = note_hospital_stay_days
            elif isinstance(hospital_stay_days, str) and hospital_stay_days.isdigit():
                hospital_stay_days = int(hospital_stay_days)
            elif isinstance(hospital_stay_days, (int, float)):
                hospital_stay_days = int(hospital_stay_days)
            else:
                hospital_stay_days = None

            medication_count = validated_data.get("medication_count")
            if medication_count is None or medication_count == "":
                note_medication_count = self._extract_medication_count(clinical_note)
                if note_medication_count is not None:
                    medication_count = note_medication_count
                elif normalized_medications:
                    medication_count = len(normalized_medications)
            elif isinstance(medication_count, str) and medication_count.isdigit():
                medication_count = int(medication_count)
            elif isinstance(medication_count, (int, float)):
                medication_count = int(medication_count)
            else:
                medication_count = None

            condition_status = validated_data.get("condition_status") or "Unknown"
            note_condition_status = self._extract_condition_status(clinical_note)
            if note_condition_status:
                condition_status = note_condition_status

            features = ClinicalFeatures(
                age_group=str(validated_data.get("age_group", "Unknown")),
                symptoms=normalized_symptoms,
                medications=normalized_medications,
                medication_count=medication_count,
                condition_status=condition_status,
                hospital_stay_days=hospital_stay_days,
                glucose_status=validated_data.get("glucose_status", "Unknown")
            )

            # Generate recommendations (this part uses the LLM again)
            recommendations_str = self._generate_recommendations(
                clinical_note,
                features,
                context_text=context_text,
                readmission_risk=readmission_prediction,
            )

            return ClinicalDecisionReport(
                features=features,
                readmission_risk=readmission_prediction,
                recommendations=recommendations_str
            )
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return None

    def _llm_parsing(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses the LLM to parse the clinical note into the extraction schema.
        """
        prompt = f"""
        You are a highly specialized clinical data extraction agent. 
        Your task is to extract specific clinical entities from the provided clinical note and optional medical context.
        
        Instructions:
        1. Extract the following fields: {self.extraction_schema}
        2. Use the clinical note as the primary source. Use the medical context only to clarify terminology or confirm explicit details.
        3. Do not infer symptoms, conditions, or glucose status from external medical knowledge.
        4. Interpret mixed English/Hindi patient text, but do not invent new symptoms, medication names, or condition details.
        5. If a value is not stated, use null or an empty list.
        6. Output the result as a valid JSON object only.
        
        Clinical Note and Context:
        {text}
        
        JSON Output:
        """
        response_json = self.llm.generate_structured_json(prompt, self.extraction_schema)
        if not response_json:
            return None
        return response_json

    def _reflect_on_extraction(self, original_note: str, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        A secondary LLM call to validate the extracted data against the original note.
        """
        prompt = f"""
        You are a clinical auditor. Your task is to verify if the extracted information matches the original clinical note.
        
        Original Note: {original_note}
        Extracted Data: {json.dumps(extracted_data)}
        
        Instructions:
        1. Compare the extracted data against the original note.
        2. Only change a field if the note clearly contradicts it or the field is missing.
        3. If a symptom or condition is not explicitly mentioned, remove it or return an empty list.
        4. Do not invent or replace values based on general knowledge or retrieved medical context.
        5. Output the result as a valid JSON object only.
        
        JSON Output:
        """
        try:
            corrected_json = self.llm.generate_structured_json(prompt, self.extraction_schema)
            return corrected_json
        except Exception:
            return None

    def _build_fallback_recommendations(self, features: ClinicalFeatures, readmission_risk: Optional[bool] = None) -> str:
        """
        Builds a concise, feature-grounded recommendation string when the LLM output is generic or malformed.
        """
        glucose_status = features.glucose_status or "Unknown"
        symptoms = features.symptoms or []
        medications = features.medications or []
        age_group = features.age_group or "Unknown"

        symptom_summary = ", ".join(symptoms[:3]) if symptoms else "the reported symptoms"
        medication_summary = ", ".join(medications[:3]) if medications else "no medications listed"

        urgency = "high" if readmission_risk is True else ("medium" if readmission_risk is False else "medium")
        if glucose_status.lower() in {"hyperglycemia", "hypoglycemia", "high", "low"}:
            base = f"Monitor {glucose_status.lower()} closely and review the patient's symptoms ({symptom_summary})."
        else:
            base = f"Continue routine monitoring and review the patient's symptoms ({symptom_summary})."

        if medications:
            base += f" Reassess medications ({medication_summary}) and confirm adherence with the care team."
        else:
            base += " Reconfirm medication history and update the care plan as needed."

        if readmission_risk is True:
            base += f" Because the risk is elevated for this {age_group} patient, prioritize timely follow-up and escalation if symptoms worsen."
        elif readmission_risk is False:
            base += f" For this {age_group} patient, routine follow-up is appropriate unless symptoms progress."
        else:
            base += f" For this {age_group} patient, ensure close follow-up and reassess if symptoms worsen."

        return base.strip()

    def _generate_recommendations(self, clinical_note: str, features: ClinicalFeatures, context_text: str = "", readmission_risk: Optional[bool] = None) -> str:
        """
        Generates clinical recommendations based on the extracted features and note.
        """
        risk_str = "High" if readmission_risk is True else ("Low" if readmission_risk is False else "Unknown")
        context_block = f"\nMedical Context: {context_text}" if context_text.strip() else ""
        prompt = f"""
        You are a clinical decision support system.
        Write a concise recommendation summary based ONLY on the provided clinical note, extracted features, and the optional medical context.
        Return 2-3 sentences maximum. Do not repeat the prompt instructions or include generic template text.

        Clinical Note: {clinical_note}
        Extracted Features: {features.json()}
        Known Readmission Risk: {risk_str}{context_block}

        Focus on monitoring, medication review, follow-up, and escalation if symptoms worsen.
        """
        response = self.llm.generate_text(prompt)
        if not response:
            return self._build_fallback_recommendations(features, readmission_risk)

        normalized = response.strip()
        low_quality_markers = [
            "format your response",
            "what is the diagnosis and treatment",
            "template",
            "professional text summary",
            "clinical decision support system",
        ]
        if any(marker in normalized.lower() for marker in low_quality_markers):
            return self._build_fallback_recommendations(features, readmission_risk)

        return normalized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clinical Orchestrator Agent with custom input.")
    parser.add_argument("--note", type=str, help="The clinical note to process.")
    parser.add_argument("--high-risk", action="store_true", help="Set prediction to True (High Risk).")
    parser.add_argument("--low-risk", action="store_true", help="Set prediction to False (Low Risk).")
    args = parser.parse_args()

    # If no arguments are provided, prompt the user for input
    if args.note is None:
        print("--- Clinical Agent Input Mode ---")
        note = input("Enter the clinical note: ")
        
        # Determine risk status based on input; allow unknown if not specified.
        risk_input = input("Is this a high risk case? (y/n/u for unknown, default 'u'): ").strip().lower()
        if risk_input == 'y':
            high_risk = True
        elif risk_input == 'n':
            high_risk = False
        else:
            high_risk = None
    else:
        note = args.note
        # If note is provided via CLI, determine risk only if specified.
        # If neither flag is set, leave the risk unknown.
        if args.high_risk:
            high_risk = True
        elif args.low_risk:
            high_risk = False
        else:
            high_risk = None

    # Dependency Injection Setup
    retriever = RAGRetriever()
    retriever.load()
    
    llm_interface = LLMInterface(provider=OllamaProvider())

    # Initialize Agent
    agent = ClinicalOrchestratorAgent(retriever=retriever, llm=llm_interface)

    logger.info("--- Starting Clinical Decision Support Process ---")
    report = agent.orchestrate(note, high_risk)

    if report:
        print("\n--- FINAL CLINICAL DECISION REPORT ---")
        print(report.model_dump_json(indent=2))
    else:
        logger.error("Failed to generate report.")
