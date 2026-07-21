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
        # Use a simplified schema for the prompt to avoid truncation and confusion
        self.extraction_schema = {
            "age_group": "string",
            "symptoms": "list of strings",
            "medications": "list of strings",
            "hospital_stay_days": "integer",
            "glucose_status": "string"
        }
        self.recommendation_schema = {
            "actionable_steps": "list of strings",
            "urgency_level": "string (Low, Medium, High)",
            "clinical_rationale": "string"
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

        except (ConnectionError, ValueError, RuntimeError) as e:
            logger.error(f"Orchestration failed: {e}")
            return None

    def _llm_parsing(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses the LLMInterface to extract structured features from the augmented text.
        """
        prompt = f"Extract clinical features from the following text:\n\n{text}"
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
            f"If the risk is HIGH, focus on urgent preventative interventions to avoid readmission. "
            f"If the risk is LOW, focus on maintenance and long-term monitoring.\n"
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
    # Dependency Injection Setup
    retriever = RAGRetriever()
    retriever.load()
    
    llm_interface = LLMInterface(provider=OllamaProvider())

    # Initialize Agent
    agent = ClinicalOrchestratorAgent(retriever=retriever, llm=llm_interface)

    # Test with a sample Hinglish note and a simulated prediction
    sample_note = "Patient age [0-10) presented with high sugar issues. Bukhar and weakness reported. Time in hospital: 4 days."
    simulated_prediction = True # Simulate that XGBoost predicted a high risk

    logger.info("--- Starting Clinical Decision Support Process ---")
    report = agent.orchestrate(sample_note, simulated_prediction)
    
    if report:
        logger.info(f"Final Clinical Decision Report:\n{report.model_dump_json(indent=2)}")
    else:
        logger.error("No report obtained.")
