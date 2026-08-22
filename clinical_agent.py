import asyncio
import argparse
import sys
import json
import re
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
from loguru import logger
from llm_providers import OllamaProvider
from rag_retriever import RAGRetriever
from llm_interface import LLMInterface
from config import config

class Symptom(BaseModel):
    """Represents a clinical symptom and its negation status."""
    name: str = Field(..., description="The name of the symptom (e.g., 'fever', 'fatigue').")
    is_negated: bool = Field(..., description="True if the symptom is explicitly denied or absent in the note (e.g., 'no fever'), False otherwise.")

class ClinicalFeatures(BaseModel):
    """Structured clinical features extracted from a clinical note."""
    age_group: str = Field(..., description="The age group of the patient (e.g., '[0-10)', 'age 25').")
    symptoms: List[Symptom] = Field(default_factory=list, description="List of reported symptoms with their negation status.")
    medications: List[str] = Field(default_factory=list, description="List of medications the patient is taking.")
    medication_count: Optional[int] = Field(None, description="The number of medications explicitly stated in the note.")
    condition_status: str = Field("Unknown", description="The patient's condition status as explicitly reported in the note.")
    hospital_stay_days: Optional[int] = Field(None, description="Number of days the patient stayed in the hospital.")
    glucose_status: str = Field("Unknown", description="The glucose status (e.g., 'Hyperglycemia', 'Hypoglycemia', 'Unknown').")

class ClinicalDecisionReport(BaseModel):
    """Clinical report containing features, prediction, and recommendations."""
    features: ClinicalFeatures
    readmission_risk: Optional[bool]
    recommendations: str

class ClinicalOrchestratorAgent:
    def __init__(self, retriever: RAGRetriever, llm: LLMInterface):
        """Initializes the Orchestrator with injected dependencies."""
        self.retriever = retriever
        self.llm = llm

        self.extraction_schema = {
            "age_group": "One word only: 'Adult', 'Pediatric', 'Geriatric', or 'Unknown'.",
            "symptoms": "Strict array format only: [{'name': 'symptom', 'is_negated': true/false}].",
            "medications": "Flat array of medication name strings only.",
            "medication_count": "Single integer value or null.",
            "condition_status": "One word only: 'Stable', 'Unstable', 'Improving', 'Worsening', or 'Unknown'.",
            "hospital_stay_days": "Single integer value or null.",
            "glucose_status": "One word only: 'Hyperglycemia', 'Hypoglycemia', 'Normoglycemia', or 'Unknown'."
        }

        self.recommendation_schema = {
            "actionable_steps": "A list of strings, where each string is a clear, professional medical instruction.",
            "urgency_level": "A single word: 'Low', 'Medium', or 'High'.",
            "clinical_rationale": "A brief, professional explanation for the recommendations based on the clinical data."
        }

    def _extract_hospital_stay_days(self, clinical_note: str) -> Optional[int]:
        """Extracts hospital stay days from note text if present."""
        patterns = [
            r'hospital\s+stay\s*[:=]\s*(\d+)',
            r'hospital\s+stay\s*(?:was|is|duration|for)?\s*[:=]?\s*(\d+)',
            r'admitted\s+for\s+(\d+)\s+days',
            r'\b(\d+)\s+days?\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, clinical_note, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_medication_count(self, clinical_note: str) -> Optional[int]:
        """Extracts medication count from note text."""
        patterns = [
            r'(?:number of medications|total medications|prescribed medications|medications)\s*[:=]\s*(\d+)',
            r'took\s+(\d+)\s+medications',
            r'\b(\d+)\s+medications\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, clinical_note, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_condition_status(self, clinical_note: str) -> Optional[str]:
        """Extracts condition status phrases from note."""
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
        Merges audited data from the clinical reflector. 
        Gives priority to reflector corrections for symptoms, negations, and clinical status.
        """
        if not isinstance(extracted_data, dict): 
            return reflected_data or {}
        if not isinstance(reflected_data, dict): 
            return extracted_data
        
        merged = extracted_data.copy()
        template_stop_tokens = ["only", "strict", "format", "array", "string", "instruction", "schema"]

        for key, ref_val in reflected_data.items():
            orig_val = str(extracted_data.get(key, '')).lower()
            if any(indicator in orig_val for indicator in template_stop_tokens) or not extracted_data.get(key):
                merged[key] = ref_val
            elif ref_val is not None and ref_val != "" and ref_val != []:
                merged[key] = ref_val

        return merged

    async def _llm_parsing(self, text: str) -> Optional[Dict[str, Any]]:
        """Parses clinical note into extraction schema."""
        prompt = f"""
        You are a clinical data extraction agent. Extract specific clinical entities from the provided clinical note and optional medical context.
        Instructions:
        1. Extract the following fields: {self.extraction_schema}
        2. Use the clinical note as the primary source. Use the medical context only to clarify terminology or confirm explicit details.
        3. Do not infer symptoms, conditions, or glucose status from external medical knowledge.
        4. Interpret mixed English/Hindi patient text, but do not invent new symptoms, medication names, or condition details.
        5. If a value is not stated, use null or an empty list.
        6. Output the result as a valid JSON object only.
        Clinical Note and Context: {text}
        JSON Output:
        """
        response_json = await self.llm.generate_structured_json(prompt, self.extraction_schema)
        return response_json if response_json else None

    async def _reflect_on_extraction(self, original_note: str, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Verifies extracted data against original note."""
        prompt = f"""
        You are a clinical auditor. Verify if the extracted information matches the original clinical note.
        Instructions:
        1. Compare the extracted data against the original note.
        2. Only change a field if the note clearly contradicts it or the field is missing.
        3. If a symptom or condition is not explicitly mentioned, remove it or return an empty list.
        4. Do not invent or replace values based on general knowledge or retrieved medical context.
        5. Output the result as a valid JSON object only.
        Original Note and Extracted Data to audit:
        Original Note: {original_note}
        Extracted Data: {json.dumps(extracted_data)}
        JSON Output:
        """
        try:
            return await self.llm.generate_structured_json(prompt, self.extraction_schema)
        except (KeyError, ValueError, TypeError):
            return None

    def _build_fallback_recommendations(self, features: ClinicalFeatures, readmission_risk: Optional[bool] = None) -> str:
        """Fallback rule-based recommendation generator."""
        glucose_status = features.glucose_status or "Unknown"
        symptoms = features.symptoms or []
        medications = features.medications or []
        age_group = features.age_group or "Unknown"
        
        symptom_summary = ", ".join(s.name for s in symptoms[:3]) if symptoms else "the reported symptoms"
        medication_summary = ", ".join(medications[:3]) if medications else "no medications listed"
        
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

    async def _generate_recommendations(self, clinical_note: str, features: ClinicalFeatures, context_text: str = "", readmission_risk: Optional[bool] = None) -> str:
        """ 
        Generates clinical recommendations based on extracted features and context.
        """
        risk_str = "High" if readmission_risk is True else ("Low" if readmission_risk is False else "Unknown")
        context_block = f"\nMedical Context: {context_text}" if context_text.strip() else ""
        prompt = f"""
        Write a concise recommendation summary (2-3 sentences max) based ONLY on this data. Do not use JSON formatting.
        Known Readmission Risk: {risk_str}{context_block}
        Clinical Note: {clinical_note}
        Extracted Features: {features.model_dump_json()}
        Focus on monitoring, medication review, follow-up and escalation.
        """
        try:
            response = await self.llm.generate_text(prompt)
                
            if not response:
                return self._build_fallback_recommendations(features, readmission_risk)
            
            normalized = response.strip()
            boilerplate_exclusions = ["format your response", "template", "clinical decision support system"]
            if any(marker in normalized.lower() for marker in boilerplate_exclusions):
                return self._build_fallback_recommendations(features, readmission_risk)
            return normalized
        except (KeyError, ValueError, TypeError):
            return self._build_fallback_recommendations(features, readmission_risk)

    async def orchestrate(
        self, 
        clinical_note: str, 
        readmission_prediction: Optional[bool] = None, 
        skip_reflection: bool = True,
        generate_recs: bool = False
    ) -> Optional[ClinicalDecisionReport]:
        """ Orchestrates RAG and LLM parsing with zero-crash fallbacks. """
        try:
            context_docs = await self.retriever.retrieve(clinical_note, k=config.RETRIEVAL_K)
            context_text = "\n".join(str(doc) for doc in context_docs)
            note_with_context = f"Clinical Note: {clinical_note}\n\nMedical Context: {context_text}"
            
            extracted_data = await self._llm_parsing(note_with_context)
            if not isinstance(extracted_data, dict):
                logger.warning("LLM JSON parsing failed; falling back to deterministic regex extraction.")
                extracted_data = {}

            if not skip_reflection and extracted_data:
                validated_data = await self._reflect_on_extraction(clinical_note, extracted_data)
                if validated_data:
                    extracted_data = self._merge_reflected_data(extracted_data, validated_data)
            hospital_stay_days = extracted_data.get("hospital_stay_days") or self._extract_hospital_stay_days(clinical_note)
            medication_count = extracted_data.get("medication_count") or self._extract_medication_count(clinical_note)
            condition_status = self._extract_condition_status(clinical_note) or extracted_data.get("condition_status", "Unknown")

            medications_raw = extracted_data.get("medications", [])
            normalized_medications = []
            if isinstance(medications_raw, list):
                for item in medications_raw:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("medication")
                        if name: normalized_medications.append(str(name).strip())
                    elif isinstance(item, str) and item.strip():
                        normalized_medications.append(item.strip())
            elif isinstance(medications_raw, str) and medications_raw.strip():
                normalized_medications = [medications_raw.strip()]

            symptoms_raw = extracted_data.get("symptoms", [])
            normalized_symptoms = []
            neg_pattern = r'\b(nahi|nahin|na|no|not|absent|denied|denies|koi nahi|bilkul nahi)\b'
            breakout_markers = r"(\{|\[|\}|\]|:|'|\"|strict_array|format_only|is_negated|instruction)"

            if isinstance(symptoms_raw, list):
                for s in symptoms_raw:
                    if isinstance(s, dict):
                        name_str = str(s.get("name") or s.get("symptom") or "").strip()
                        if re.search(breakout_markers, name_str.lower()): continue
                        is_neg = s.get("is_negated") or s.get("negated") or False
                        if re.search(neg_pattern, name_str.lower()):
                            is_neg = True
                            name_str = re.sub(neg_pattern, "", name_str.lower()).strip()
                        name_str = name_str.replace("  ", " ")
                        if name_str: normalized_symptoms.append(Symptom(name=name_str, is_negated=bool(is_neg)))
                    elif isinstance(s, str) and s.strip():
                        name_str = s.strip()
                        if re.search(breakout_markers, name_str.lower()): continue
                        is_neg = False
                        if re.search(neg_pattern, name_str.lower()):
                            is_neg = True
                            name_str = re.sub(neg_pattern, "", name_str.lower()).strip()
                        name_str = name_str.replace("  ", " ")
                        if name_str: normalized_symptoms.append(Symptom(name=name_str, is_negated=is_neg))

            features = ClinicalFeatures(
                age_group=str(extracted_data.get("age_group", "Unknown")),
                symptoms=normalized_symptoms,
                medications=normalized_medications,
                medication_count=int(medication_count) if str(medication_count).isdigit() else None,
                condition_status=str(condition_status),
                hospital_stay_days=int(hospital_stay_days) if str(hospital_stay_days).isdigit() else None,
                glucose_status=str(extracted_data.get("glucose_status", "Unknown"))
            )

            recommendations_str = ""
            if generate_recs:
                recommendations_str = await self._generate_recommendations(
                    clinical_note, features, context_text=context_text, readmission_risk=readmission_prediction
                )

            return ClinicalDecisionReport(
                features=features,
                readmission_risk=readmission_prediction,
                recommendations=recommendations_str
            )
        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.error(f"Orchestration critical error: {e}")
            return ClinicalDecisionReport(
                features=ClinicalFeatures(
                    age_group="Unknown", symptoms=[], medications=[], medication_count=None,
                    condition_status="Unknown", hospital_stay_days=None, glucose_status="Unknown"
                ),
                readmission_risk=readmission_prediction,
                recommendations=""
            )

async def main_async():
    parser = argparse.ArgumentParser(description="Run Clinical Orchestrator Agent with custom input.")
    parser.add_argument("--note", type=str, help="The clinical note to process.")
    parser.add_argument("--high-risk", action="store_true", help="Set prediction to True (High Risk).")
    parser.add_argument("--low-risk", action="store_true", help="Set prediction to False (Low Risk).")
    args = parser.parse_args()

    if args.note is None:
        print("--- Clinical Agent Input Mode ---")
        note = input("Enter the clinical note: ")
        risk_input = input("Is this a high risk case? (y/n/u for unknown, default 'u'): ").strip().lower()
        high_risk = True if risk_input == 'y' else (False if risk_input == 'n' else None)
    else:
        note = args.note
        high_risk = True if args.high_risk else (False if args.low_risk else None)

    retriever = RAGRetriever()
    retriever.load()
    provider = OllamaProvider()
    llm_interface = LLMInterface(provider=provider)
    
    agent = ClinicalOrchestratorAgent(retriever=retriever, llm=llm_interface)
    logger.info("--- Starting Clinical Decision Support Process ---")
    
    report = await agent.orchestrate(note, high_risk)
    if report:
        print("\n--- FINAL CLINICAL DECISION REPORT ---")
        print(report.model_dump_json(indent=2))
    else:
        logger.error("Failed to generate report.")
        
    await provider.close()
    retriever.close()

if __name__ == "__main__":
    asyncio.run(main_async())