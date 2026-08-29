import random
from pathlib import Path
from loguru import logger
import pandas as pd
from config import config

def get_grounded_symptom_pools(row: pd.Series):
    """
    Selects affirmed and negated Hinglish symptom candidates conditioned on
    the patient's actual primary diagnosis category and clinical features.
    """
    diag_cat = str(row.get('diag_1_category', 'Other'))
    a1c = str(row.get('A1Cresult', 'None'))
    insulin = str(row.get('insulin', 'No')).lower()
    
    affirmed_pool = []
    negated_pool = []
    if diag_cat == 'Circulatory':
        affirmed_pool.extend([
            "Chest discomfort aur saans lene me dikkat (dyspnea) notice ki gayi.",
            "Patient complains of palpitations aur ghabrahat during exertion.",
            "Mild swelling (sojan) observed in lower extremities."
        ])
        negated_pool.extend([
            "Severe radiating chest pain bilkul nahi hai.",
            "Syncope ya loss of consciousness completely absent.",
            "Cough ya sputum production nahi hai."
        ])
    elif diag_cat == 'Diabetes':
        affirmed_pool.extend([
            "Patient ko bahut zyada sugar issues aur weakness reported hai.",
            "Frequent urination aur bahut zyada thakan (fatigue) mehsoos ho rahi hai.",
            "Complains of dhundla dikhna (blurred vision) over past few weeks."
        ])
        negated_pool.extend([
            "Diabetic ketoacidosis (DKA) ke lakshan nahi hain.",
            "Persistent vomiting ya nausea nahi hai.",
            "Foot ulcers ya numbness denied by patient."
        ])
    elif diag_cat == 'Respiratory':
        affirmed_pool.extend([
            "Severe cough aur saans phulna (dyspnea) reported hai.",
            "Chest congestion aur wheezing notice ki gayi."
        ])
        negated_pool.extend([
            "Hemoptysis (blood in cough) bilkul nahi hai.",
            "High grade tez bukhar (fever) absent hai.",
            "Chest pain denied by the patient."
        ])
    elif diag_cat == 'Digestive':
        affirmed_pool.extend([
            "Pet dard (abdominal pain) aur indigestion ki complaint hai.",
            "Loss of appetite aur mild weakness reported."
        ])
        negated_pool.extend([
            "Hematemesis ya severe vomiting nahi hai.",
            "Jaundice ke signs bilkul absent hain."
        ])
    else:
        affirmed_pool.extend([
            "General thakan (fatigue) aur body dard (pain) reported hai.",
            "Ch चक्कर (dizziness) reported during morning admission.",
            "Mild weakness and low energy observed."
        ])
        negated_pool.extend([
            "Tez bukhar (high fever) nahi hai.",
            "Chest pain completely absent.",
            "Koi saans phulna ya shortness of breath nahi hai."
        ])
    if a1c in ['>8', '>7']:
        affirmed_pool.append(f"Recent HbA1c result was elevated ({a1c}), indicating uncontrolled glycemia.")
    elif a1c == 'Norm':
        negated_pool.append("Severe long-term HbA1c elevation absent (Norm).")

    if insulin in ['steady', 'up', 'down']:
        affirmed_pool.append(f"Patient is currently on active insulin protocol ({insulin}).")
    else:
        negated_pool.append("No active exogenous insulin regimen prescribed.")

    return affirmed_pool, negated_pool

def generate_hinglish_note(row: pd.Series) -> str:
    """
    Constructs a diverse, coherent Hinglish clinical note for a single patient record.
    """
    age = row.get('age', row.get('age_group', 'Unknown'))
    stay = row.get('time_in_hospital', 1)
    num_labs = row.get('num_lab_procedures', 0)
    num_meds = row.get('total_medications_count', row.get('num_medications', 0))
    has_meds = row.get('diabetesMed') == 'Yes' or row.get('diabetesMed_binary') == 1
    med_status = "active diabetes medications" if has_meds else "no active diabetes meds"
    diag_cat = row.get('diag_1_category', 'Clinical')

    affirmed_pool, negated_pool = get_grounded_symptom_pools(row)

    # Sample without replacement for unique narrative variations
    k_affirmed = min(2, len(affirmed_pool))
    k_negated = min(1, len(negated_pool))

    selected_affirmed = random.sample(affirmed_pool, k=k_affirmed)
    selected_negated = random.sample(negated_pool, k=k_negated)

    narrative_phrases = selected_affirmed + selected_negated
    random.shuffle(narrative_phrases)

    p1 = narrative_phrases[0] if len(narrative_phrases) > 0 else ""
    p2 = narrative_phrases[1] if len(narrative_phrases) > 1 else ""
    p3 = narrative_phrases[2] if len(narrative_phrases) > 2 else ""

    templates = [
        f"Patient age {age} admitted for {stay} days. Diagnosis category: {diag_cat}. {p1} {p2} {p3} Number of medications: {num_meds}. Total lab procedures: {num_labs}.",
        f"Clinical note: Age {age}. Patient is on {med_status}. {p1} Hospital stay duration: {stay} days. {p2} {p3} Lab tests completed: {num_labs}.",
        f"Patient {age} years old presenting with {diag_cat} issues. {p1} {p2} Prescribed medications: {num_meds}. Hospital stay: {stay} days. {p3}",
        f"Summary: Age {age} patient with {med_status}. {p1} {p2} Stay was {stay} days. Diagnostic tests: {num_labs}. {p3}"
    ]

    return random.choice(templates).strip()

def main():
    tasks = [
        (config.TRAIN_DATA_PATH, config.TRAIN_WITH_NOTES_PATH),
        (config.TEST_DATA_PATH, config.TEST_WITH_NOTES_PATH)
    ]
    random.seed(42)
    
    for input_path, output_path in tasks:
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}. Please run preprocess_data.py first.")
            continue
            
        logger.info(f"Loading split dataset from: {input_path}")
        df = pd.read_csv(input_path)
        
        logger.info(f"Generating grounded Hinglish clinical notes for {input_path.name} ({len(df)} records)...")
        df['clinical_note'] = [generate_hinglish_note(row) for _, row in df.iterrows()]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"Saved notes dataset to: {output_path}")
        sample = df['clinical_note'].iloc[0]
        logger.info(f"Sample generated note:\n{sample}\n")

if __name__ == "__main__":
    main()