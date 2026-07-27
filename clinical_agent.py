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
        # IMPROVED: More descriptive schema to guide the LLM and prevent hallucinations
        self.extraction_schema = {
            "age_group": "The patient's age as a number, range (e.g., '30-40'), or group (e.g., 'adult'). If not found, use 'Unknown'.",
            "symptoms": "A list of symptoms explicitly mentioned in the text. If none are mentioned, return an empty list [].",
            "medications": "A list of medications explicitly mentioned in the text. If none are mentioned, return an empty list [].",
            "hospital_stay_days": "The number of days spent in the hospital as an integer. If not mentioned, use null.",
            "glucose_status": "The glucose status extracted from text (e.g., 'Hyperglycemia', 'Hypoglycemia', 'Normal', 'Unknown')."
        }
        self.recommendation_schema = {
            "actionable_steps": "A list of human-readable, professional medical instructions (e.g., 'Monitor blood glucose levels daily').",
            "urgency_level": "A single word: 'Low', 'Medium', or 'High'.",
            "clinical_rationale": "A brief, professional explanation for the recommendations based on the clinical data."
        }

    def orchestrate(self, clinical_note: str, readmission_prediction: bool) -> Optional[ClinicalDecisionReport]:
        """
        Orchestrates the R/RAG and LLM pipeline to extract features, 
        incorporate prediction, and generate clinical recommendations.
        """
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

            # Validate the extracted data using the Pydantic model
            features = ClinicalFeatures(**extracted_data)
            logger.info("Generating clinical recommendations based on prediction...")
            recommendations = self._generate_recommendations(features, readmission_prediction, context_text)

            logger.success("Successfully generated complete Clinical Decision Report.")
            
            return ClinicalDecisionReport(
                features=features,
                readmission_risk=readmission_prediction,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return None

    def _llm_parsing(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses the LLMInterface to extract structured features from the augmented text.
        """
        # IMPROVED: Added strict extraction instructions to prevent hallucination
        prompt = (
            "You are a medical data extraction assistant. Your task is to extract information "
            "STRICTLY from the provided text. Do not infer symptoms or medications that are not "
            "explicitly stated. If a piece of information is missing, follow the schema instructions.\n\n"
            f"Text to process:\n{text}"
        )
        return self.llm.generate_structured_json(prompt, self.extraction_schema)

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
        except Exception as e:
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
