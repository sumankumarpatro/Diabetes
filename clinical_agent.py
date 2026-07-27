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
    hospital_stay_days: Optional[int] = Field(None, description="Number of days the patient stayed in the hospital.")
    glucose_status: str = Field(..., description="The glucose status (e.g., 'Hyperglycemia', 'Hypoglycemia', 'Unknown').")

class ClinicalDecisionReport(BaseModel):
    """
    The final output of the Clinical Orchestrator, containing features, prediction, and recommendations.
    """
    features: ClinicalFeatures
    readmission_risk: bool
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
            "hospital_stay_days": "An integer or null.",
            "glucose_status": "The glucose status (e.g., 'Hyperglycemia', 'Hypoglycemia', 'Normal', 'Unknown')."
        }
        self.recommendation_schema = {
            "actionable_steps": "A list of strings, where each string is a clear, professional medical instruction (e.g., 'Monitor blood glucose levels daily').",
            "urgency_level": "A single word: 'Low', 'Medium', or 'High'.",
            "clinical_rationale": "A brief, professional explanation for the recommendations based on the clinical data."
        }

    def orchestrate(self, clinical_note: str, readmission_prediction: bool) -> Optional[ClinicalDecisionReport]:
        """
        Orchestrates the R/RAG and LLM pipeline to extract features, 
        incorporate prediction, and generate clinical recommendations.
        """
        print(f"DEBUG: Orchestrate called with note: {clinical_note[:50]}...")
        logger.info(f"Received Clinical Note: {clinical_note[:100]}...")
        
        try:
            logger.info("Retrieving medical context...")
            context_docs = self.retriever.retrieve(clinical_note, k=config.RETRIEVAL_K)
            context_text = "\n".join(context_docs)
            logger.debug(f"Retrieved Context: {context_text[:200]}...")
            augmented_text = f"Clinical Note: {clinical_note}\n\nMedical Context: {context_text}"
            logger.info("Extracting features via LLM...")
            extracted_data = self._llm_parsing(augmented_text)
            
            if not extracted_data:
                logger.warning("Failed to extract features from note.")
                return None
            logger.info("Reflecting on extraction accuracy...")
            validated_data = self._reflect_on_extraction(clinical_note, extracted_data)
            
            if not validated_data:
                logger.warning("Reflector failed to validate extraction. Falling back to original data.")
                validated_data = extracted_data
            else:
                logger.success("Reflector validated/corrected the extraction.")

            # Validate the extracted data using the Pydantic model
            features = ClinicalFeatures(**validated_data)
            logger.info("Generating clinical recommendations based on prediction...")
            recommendations = self._generate_recommendations(features, readmission_prediction, context_text)

            logger.success("Successfully generated complete Clinical Decision Report.")
            
            return ClinicalDecisionReport(
                features=features,
                readmission_risk=readmission_prediction,
                recommendations=recommendations
            )

        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.error(f"Orchestration failed: {e}")
            return None

    def _llm_parsing(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses the LLMInterface to extract structured features from the augmented text.
        """
        # IMPROVED: Added explicit formatting rules and strict constraints to prevent hallucinations
        prompt = (
            "You are a medical data extraction assistant. Your task is to extract information "
            "STRICTLY from the provided text. Do not infer symptoms or medications that are not "
            "explicitly stated. If a piece of information is missing, follow the schema instructions.\n\n"
            "STRICT CONSTRAINTS:\n"
            "1. NO INFERENCE: Do not assume any symptoms, medications, or age if not explicitly written.\n"
            "2. NO PLACEHOLDERS: NEVER return the literal strings 'string', 'list of ' or 'integer'.\n"
            "3. MISSING DATA: If a piece of information is missing, use 'Unknown', an empty list [], or null.\n"
            "4. ACCURACY: Every extracted value must be traceable to the provided text.\n\n"
            "FORMATTING RULES:\n"
            "1. Replace all schema placeholders with ACTUAL data found in the text.\n"
            "2. Ensure the output is a valid JSON object.\n\n"
            f"Text to process:\n{text}"
        )
        return self.llm.generate_structured_json(prompt, self.extraction_schema)

    def _reflect_on_extraction(self, original_note: str, extracted_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        The 'Reflector' step: A second LLM call to verify that the extracted data 
        is strictly present in the original note and contains no hallucinations.
        Uses a Chain-of-Thought approach by requiring an 'audit_log' in the schema.
        """
        # Define a schema for the auditor that includes an audit log for Chain-of-Thought
        audit_schema = self.extraction_schema.copy()
        audit_schema["audit_log"] = "A list of strings where you document your verification process for each field (e.g., 'Verified age: 30 is in text', 'Removed symptom: headache because it is not in text')."

        prompt = (
            "You are a strict medical auditor. Your ONLY job is to remove hallucinations.\n\n"
            "AUDIT PROCESS:\n"
            "1. Review the 'EXTRACTED DATA' field by field.\n"
            "2. For each field, check if the value is explicitly stated in the 'ORIGINAL NOTE'.\n"
            "3. If a value is NOT in the text, you MUST remove it from the JSON and note it in the 'audit_log'.\n"
            "4. If a value IS in the text, you MUST confirm it in the 'audit_log'.\n\n"
            "AUDIT RULE:\n"
            "Compare the 'EXTRACTED DATA' against the 'ORIGINAL NOTE'. If a symptom, medication, "
            "or age is mentioned in the JSON but NOT in the text, you MUST remove it from the JSON.\n\n"
            f"ORIGINAL NOTE: {original_note}\n\n"
            f"EXTRACTED DATA: {json.dumps(extracted_data)}\n\n"
            "IMPORTANT: Ensure the output follows the correct format. Do not use 'string' or 'list of strings'.\n"
            "Respond ONLY with the corrected JSON object containing the updated data and your 'audit_log'."
        )
        
        try:
            audit_result = self.llm.generate_structured_json(prompt, audit_schema)
            if not audit_result:
                return None
            
            # Extract only the original fields from the audit result
            cleaned_data = {key: audit_result[key] for key in self.extraction_schema if key in audit_result}
            
            # Log the audit process for debugging/visibility
            if "audit_log" in audit_result:
                logger.info(f"Reflector Audit Log: {' | '.join(audit_result['audit_log'])}")
                
            return cleaned_data
        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.error(f"Reflector error: {e}")
            return None

    def _generate_recommendations(self, features: ClinicalFeatures, prediction: bool, context: str) -> str:
        """
        Uses the LLM to generate personalized clinical recommendations based on 
        extracted features, the prediction, and retrieved medical context.
        """
        risk_status = "HIGH RISK of readmission" if prediction else "LOW RISK of readmission"
        
        prompt = (
            f"Based on the following clinical data, generate personalized medical recommendations.\n\n"
            f"Patient Features: {features.model_dump_json()}\n"
            f"Prediction Status: {risk_status}\n"
            f"Medical Context: {context}\n\n"
            f"Instructions:\n"
            f"1. Use professional, natural language for all fields.\n"
            f"2. If the risk is HIGH, focus on urgent preventative interventions to avoid readmission.\n"
            f"3. If the risk is LOW, focus on maintenance and long-term monitoring.\n"
            f"4. IMPORTANT: If the patient is stable or low risk, include encouraging, generic health maintenance advice (e.g., 'Continue healthy lifestyle', 'Maintain regular checkups').\n"
            f"5. NEVER return 'No recommendations could be generated' unless there is absolutely no data to work with.\n"
            f"Respond ONLY with a valid JSON object following this schema: {json.dumps(self.recommendation_schema)}."
        )

        recommendations_json = self.llm.generate_structured_json(prompt, self.recommendation_schema)
        
        if not recommendations_json:
            return "No recommendations could be generated."

        try:
            recs = recommendations_json
            steps = "\n".join([f"- {step}" for step in recs.get("actionable_steps", [])])
            return (
                f"**Urgency: {recs.get('urgency_level', 'Unknown')}**\n\n"
                f"**Actionable Steps:**\n{steps}\n\n"
                f"**Rationale:** {recs.get('clinical_rationale', 'N/A')}"
            )
        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.error(f"Error parsing recommendations: {e}")
            return "Error generating formatted recommendations."

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
        
        # Determine risk status based on input or default to low risk if not specified
        # We'll check if the user wants to specify high risk via a prompt
        risk_input = input("Is this a high risk case? (y/n, default 'n'): ").strip().lower()
        high_risk = risk_input == 'y'
    else:
        note = args.note
        # If note is provided via CLI, we need to determine if it's high risk
        # If --high-risk is not set, but --low-risk is, it's low risk.
        # If neither is set, we'll default to low risk.
        high_risk = args.high_risk

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
