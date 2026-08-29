import argparse
from pathlib import Path
from loguru import logger
from config import config

KB_DOCUMENTS = {
    "diabetes_pathophysiology.txt": """Diabetes Mellitus Pathophysiology:
Diabetes is a group of metabolic diseases characterized by high blood glucose levels over a prolonged period. The primary issue is either insufficient insulin production (Type 1) or the body's inability to effectively use the insulin it produces (Type 2).

Key mechanisms:
- Insulin Resistance: Cells in muscles, fat, and the liver do not respond properly to insulin.
- Beta-cell Dysfunction: The pancreas fails to produce enough insulin to overcome resistance.
- Glucose Dysregulation: Resulting in hyperglycemia, which leads to long-term microvascular and macrovascular damage.""",

    "risk_factors.txt": """Diabetes Risk Factors:
- Obesity: Excess body fat, particularly abdominal fat, is a major risk factor for Type 2 diabetes.
- Physical Inactivity: A sedentary lifestyle increases insulin resistance.
- Genetics: Family history of diabetes significantly increases risk.
- Age: Risk increases as you get older, especially after age 45.
- Ethnicity: Certain populations (including African American, Hispanic/Latino, and Native American) are at higher risk.""",

    "diabetic_ketoacidosis_dka.txt": """Diabetic Ketoacidosis (DKA) Presentation:
DKA is a life-threatening complication of diabetes, often seen in Type 1.
Symptoms:
- Fruity-smelling breath (acetone).
- Kussmaul breathing (deep, rapid breathing).
- Nausea, vomiting, and abdominal pain.
- High blood glucose, ketones in urine, and metabolic acidosis.""",

    "a1c_interpretation.txt": """Diabetes Diagnostic Tests:
- HbA1c Test: Measures average blood sugar over 3 months. (>= 6.5% = Diabetes, > 8.0% = High Risk/Poor Control).
- Fasting Plasma Glucose (FPG): Measures blood sugar after fasting. (>= 126 mg/dL = Diabetes).
- Oral Glucose Tolerance Test (OGTT): Measures response to a sugar load. (>= 200 mg/dL = Diabetes).
- Random Plasma Glucose Test: High glucose levels (> 200 mg/dL) with symptoms.

Glucose Monitoring:
- Self-Monitoring of Blood Glucose (SMBG): Frequent finger-prick tests.
- Continuous Glucose Monitoring (CGM): Wearable sensor for real-time tracking.
- Importance: Essential for preventing both severe hyperglycemia and hypoglycemia.""",

    "oral_antidiabetic_drugs.txt": """Diabetes Management Strategies:
- Insulin Therapy: Essential for Type 1; used in Type 2 when oral agents fail.
- Oral Antidiabetics:
  - Biguanides (Metformin): First-line therapy to reduce hepatic gluconeogenesis.
  - Sulfonylureas (Glipizide, Glyburide): Stimulate beta-cell insulin secretion.
  - SGLT2 Inhibitors: Promote glucose excretion in urine.
  - DPP-4 Inhibitors: Increase incretin levels to enhance insulin synthesis.
- Lifestyle: Carbohydrate management, regular exercise, and weight control.""",

    "microvascular_complications.txt": """Diabetes Complications:
- Microvascular:
  - Retinopathy (eye damage, blurred vision).
  - Nephropathy (kidney damage, proteinuria).
  - Neuropathy (nerve damage, numbness, peripheral pain).
- Macrovascular:
  - Cardiovascular disease (coronary artery disease, heart attack, stroke).
  - Peripheral artery disease.
- Foot Complications: Ulcers, delayed wound healing, and potential amputation due to poor circulation and neuropathy.""",

    "clinical_indicators.txt": """Clinical Indicators in Diabetes Management:
- A1Cresult: Represents average blood glucose levels over the past 2-3 months. Values > 6.5% indicate diabetes; values > 8.0% indicate elevated readmission risk.
- max_glu_serum: The highest glucose level measured during the encounter. High levels (hyperglycemia > 200 mg/dL) increase acute metabolic complication risk.
- num_lab_procedures: Number of laboratory tests performed. Higher counts correlate with diagnostic complexity and intensive monitoring.
- Glucose Fluctuations: Monitoring glycemic volatility is critical for preventing acute DKA and hyperosmolar states.""",

    "medication_interactions.txt": """Diabetes Medication Overview:
- Metformin: Biguanide reducing hepatic glucose output. Monitored for renal impairment.
- Insulin: Essential for exogenous insulin replacement; dosage adjustments indicate disease volatility.
- Sulfonylureas (Glipizide, Glyburide, Glimepiride): Stimulate insulin release; carries hypoglycemia risk.
- Thiazolidinediones (Pioglitazone, Rosiglitazone): Improve peripheral insulin sensitivity.
- Medication Count: High concurrent medication counts (polypharmacy >= 5) indicate severe comorbid disease and drug interaction risk.""",

    "hospital_metrics.txt": """Healthcare Utilization and Complexity:
- time_in_hospital: Longer hospital stays indicate severe acute illness or complex diagnostic processes.
- number_emergency: Emergency room visits are strong predictors of chronic disease instability and high 30-day readmission risk.
- number_inpatient: Previous hospitalizations indicate severe recurrent disease decompensation.
- number_diagnoses: Higher counts of recorded comorbid diagnoses reflect elevated clinical complexity.""",

    "multilingual_clinical_glossary.txt": """Multilingual (Hinglish) Clinical Lexicon and Negation Terms:
- Bukhar: Fever / Pyrexia.
- Thakan / Kamzori: Fatigue / Generalized weakness.
- Saans phulna / Saans lene me dikkat: Dyspnea / Shortness of breath.
- Dard / Seene me dard: Pain / Chest pain / Angina.
- Ch चक्कर / Chakkar aana: Dizziness / Vertigo.
- Dhundla dikhna: Blurred vision / Visual disturbance.
- Pet dard / Ulti: Abdominal pain / Nausea / Vomiting.
- Sojan: Edema / Swelling in extremities.
- Negation terms ('nahi', 'nahin', 'na', 'absent', 'denied'): Denote confirmed absence of symptoms in clinical notes."""
}

def generate_kb(force: bool = False) -> None:
    kb_dir = config.PROJECT_ROOT / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)

    created_count, skipped_count = 0, 0

    for filename, content in KB_DOCUMENTS.items():
        target = kb_dir / filename
        if target.exists() and not force:
            logger.debug(f"skipping {filename} (already exists)")
            skipped_count += 1
            continue

        target.write_text(content.strip() + "\n", encoding="utf-8")
        logger.info(f"wrote {filename}")
        created_count += 1

    logger.success(f"knowledge base ready at {kb_dir} ({created_count} written, {skipped_count} skipped)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clinical knowledge base files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    generate_kb(force=args.force)
