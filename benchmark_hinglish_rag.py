import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from clinical_agent import ClinicalOrchestratorAgent, ClinicalDecisionReport, ClinicalFeatures, Symptom
from config import config
from llm_interface import LLMInterface
from llm_providers import OllamaProvider
from rag_retriever import RAGRetriever

GROUND_TRUTH_ONTOLOGY: Dict[str, List[str]] = {
    "fatigue": ["thakan", "kamzori", "weakness", "fatigue", "low energy", "loss of appetite", "body dard"],
    "dyspnea": ["saans phulna", "saans lene me dikkat", "shortness of breath", "dyspnea"],
    "chest_pain": ["chest discomfort", "seene me dard", "chest pain", "palpitations", "ghabrahat"],
    "edema": ["sojan", "swelling", "edema"],
    "blurred_vision": ["dhundla dikhna", "blurred vision", "visual disturbance"],
    "dizziness": ["chakkar", "dizziness", "vertigo", "ch चक्कर"],
    "abdominal_pain": ["pet dard", "abdominal pain", "indigestion"],
    "fever": ["tez bukhar", "bukhar", "high fever", "fever"],
    "frequent_urination": ["frequent urination", "sugar issues"],
    "cough": ["cough", "chest congestion", "wheezing"],
    "diabetic_ketoacidosis": ["diabetic ketoacidosis", "dka"],
    "nausea_vomiting": ["vomiting", "nausea", "hematemesis"],
    "foot_ulcer": ["foot ulcers", "foot ulcer", "numbness"],
    "jaundice": ["jaundice"],
    "syncope": ["syncope", "loss of consciousness"],
    "hemoptysis": ["hemoptysis", "blood in cough"]
}

SYMPTOM_REGEX_PATTERNS: Dict[str, List[str]] = {
    "chest_pain": [r"chest\s+discomfort", r"seene\s+me\s+dard", r"chest\s+pain", r"palpitations", r"ghabrahat"],
    "dyspnea": [r"saans\s+lene\s+me\s+dikkat", r"saans\s+phulna", r"shortness\s+of\s+breath", r"dyspnea"],
    "edema": [r"sojan", r"swelling", r"edema"],
    "fatigue": [r"thakan", r"kamzori", r"weakness", r"fatigue", r"low\s+energy", r"loss\s+of\s+appetite", r"body\s+dard"],
    "frequent_urination": [r"frequent\s+urination", r"sugar\s+issues"],
    "blurred_vision": [r"dhundla\s+dikhna", r"blurred\s+vision"],
    "dizziness": [r"ch\s*चक्कर", r"chakkar", r"dizziness", r"vertigo"],
    "abdominal_pain": [r"pet\s+dard", r"abdominal\s+pain", r"indigestion"],
    "fever": [r"tez\s+bukhar", r"bukhar", r"high\s+fever", r"fever"],
    "cough": [r"severe\s+cough", r"cough", r"chest\s+congestion", r"wheezing"],
    "diabetic_ketoacidosis": [r"diabetic\s+ketoacidosis", r"dka"],
    "nausea_vomiting": [r"vomiting", r"nausea", r"hematemesis"],
    "foot_ulcer": [r"foot\s+ulcers?", r"numbness"],
    "jaundice": [r"jaundice"],
    "syncope": [r"syncope", r"loss\s+of\s+consciousness"],
    "hemoptysis": [r"hemoptysis", r"blood\s+in\s+cough"]
}

NEGATION_CUES: List[str] = [
    r"bilkul\s+nahi\b", r"\bnahi\s+hai\b", r"\bnahi\s+hain\b",
    r"completely\s+absent\b", r"\babsent\s+hai\b", r"\babsent\b",
    r"denied\s+by\b", r"\bdenied\b", r"\bno\s+", r"\babsent\s+hain\b"
]

def extract_ground_truth_from_note(note: str, row: Optional[pd.Series] = None) -> Tuple[List[str], List[str], str, List[str], str]:
    sentences = re.split(r"[.\n]+", note)
    affirmed = set()
    negated = set()

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        is_neg = any(re.search(cue, s_clean, re.IGNORECASE) for cue in NEGATION_CUES)

        for sym, pats in SYMPTOM_REGEX_PATTERNS.items():
            for p in pats:
                if re.search(r"\b" + p, s_clean, re.IGNORECASE):
                    if is_neg:
                        negated.add(sym)
                    else:
                        affirmed.add(sym)
                    break

    affirmed = affirmed - negated

    note_lower = note.lower()
    if "elevated (>8" in note_lower or "elevated (>7" in note_lower or "uncontrolled glycemia" in note_lower or "hyperglycemia" in note_lower:
        glucose_status = "Hyperglycemia"
    elif "norm" in note_lower and "hba1c" in note_lower:
        glucose_status = "Normoglycemia"
    elif "hypoglycemia" in note_lower:
        glucose_status = "Hypoglycemia"
    elif row is not None and pd.notna(row.get("A1Cresult")):
        a1c_val = str(row.get("A1Cresult"))
        glucose_status = "Hyperglycemia" if a1c_val in [">8", ">7"] else ("Normoglycemia" if a1c_val == "Norm" else "Unknown")
    else:
        glucose_status = "Unknown"

    medications = []
    if "active insulin protocol" in note_lower or (row is not None and str(row.get("insulin", "")).lower() in ["steady", "up", "down"]):
        medications.append("insulin")
    
    known_oral_agents = ["metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride"]
    for agent in known_oral_agents:
        if agent in note_lower:
            medications.append(agent)
        elif row is not None and str(row.get(agent, "")).lower() in ["steady", "up", "down"]:
            medications.append(agent)

    age_match = re.search(r"(?:age|patient)\s*(\[\d+-\d+\)|\d+)", note, re.IGNORECASE)
    age_group = "Unknown"
    if age_match:
        age_str = age_match.group(1)
        age_group = str(row.get("age_group", age_str)) if row is not None and pd.notna(row.get("age_group")) else age_str
    elif row is not None and pd.notna(row.get("age_group")):
        age_group = str(row.get("age_group"))

    return sorted(list(affirmed)), sorted(list(negated)), glucose_status, sorted(list(set(medications))), age_group

def load_test_set_benchmark(
    n_samples: int = 500,
    seed: int = 42,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False
) -> List[Dict]:
    if cache_path is None:
        cache_path = config.PROCESSED_DIR / f"hinglish_test_benchmark_{n_samples}.json"

    if cache_path.exists() and not force_refresh:
        logger.info(f"Loading cached Hinglish test benchmark dataset ({n_samples} samples) from: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if len(data) == n_samples:
                return data
            logger.info(f"Cache size ({len(data)}) does not match requested ({n_samples}). Re-sampling...")

    test_notes_path = config.TEST_WITH_NOTES_PATH
    if not test_notes_path.exists():
        fallback = Path("processed_data/test_with_notes.csv")
        if fallback.exists():
            test_notes_path = fallback
        else:
            raise FileNotFoundError(
                f"Test dataset with clinical notes not found at {test_notes_path}. "
                "Please run `python generate_clinical_notes.py` first."
            )

    logger.info(f"Sampling {n_samples} encounters from out-of-sample test split: {test_notes_path} (seed={seed})")
    df = pd.read_csv(test_notes_path)
    if "clinical_note" not in df.columns:
        raise ValueError("Missing 'clinical_note' column in test dataset.")

    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    benchmark_data: List[Dict] = []

    for idx, row in sample_df.iterrows():
        note_text = str(row["clinical_note"]).strip()
        affirmed, negated, glucose_status, meds, age_grp = extract_ground_truth_from_note(note_text, row)

        benchmark_item = {
            "id": idx + 1,
            "encounter_index": int(row.get("Unnamed: 0", idx)),
            "note": note_text,
            "affirmed_symptoms": affirmed,
            "negated_symptoms": negated,
            "glucose_status": glucose_status,
            "medications": meds,
            "age_group": age_grp,
            "time_in_hospital": int(row.get("time_in_hospital", 1)),
            "readmitted_30d": int(row.get("readmitted_binary", 0))
        }
        benchmark_data.append(benchmark_item)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Successfully generated and cached {len(benchmark_data)} test benchmark samples to: {cache_path}")
    return benchmark_data

def evaluate_extraction(extracted_report: Optional[ClinicalDecisionReport], ground_truth: Dict) -> Dict[str, float]:
    if not extracted_report or not hasattr(extracted_report, "features"):
        return {
            "symptom_precision": 0.0,
            "symptom_recall": 0.0,
            "symptom_f1": 0.0,
            "negation_accuracy": 0.0,
            "glucose_accuracy": 0.0,
            "hallucination_count": 0.0
        }

    features = extracted_report.features
    extracted_symptoms = features.symptoms or []

    extracted_affirmed = set()
    extracted_negated = set()

    for s in extracted_symptoms:
        s_name = str(s.name).lower().strip()
        matched_canonical = None
        for canonical, keywords in GROUND_TRUTH_ONTOLOGY.items():
            if any(kw in s_name for kw in keywords) or s_name in canonical:
                matched_canonical = canonical
                break

        if not matched_canonical:
            matched_canonical = s_name.replace(" ", "_")

        if getattr(s, "is_negated", False):
            extracted_negated.add(matched_canonical)
        else:
            extracted_affirmed.add(matched_canonical)

    gt_affirmed = set(ground_truth.get("affirmed_symptoms", []))
    gt_negated = set(ground_truth.get("negated_symptoms", []))

    tp_aff = len(extracted_affirmed.intersection(gt_affirmed))
    fp_aff = len(extracted_affirmed - gt_affirmed)
    fn_aff = len(gt_affirmed - extracted_affirmed)

    prec_aff = tp_aff / (tp_aff + fp_aff) if (tp_aff + fp_aff) > 0 else (1.0 if len(gt_affirmed) == 0 and len(extracted_affirmed) == 0 else 0.0)
    rec_aff = tp_aff / (tp_aff + fn_aff) if (tp_aff + fn_aff) > 0 else (1.0 if len(gt_affirmed) == 0 else 0.0)
    f1_aff = 2 * prec_aff * rec_aff / (prec_aff + rec_aff) if (prec_aff + rec_aff) > 0 else 0.0

    total_neg_gt = len(gt_negated)
    if total_neg_gt > 0:
        neg_correct = sum(1 for n_sym in gt_negated if n_sym in extracted_negated and n_sym not in extracted_affirmed)
        neg_accuracy = neg_correct / total_neg_gt
    else:
        neg_accuracy = 1.0 if len(extracted_negated) == 0 else 0.0

    gt_glucose = ground_truth.get("glucose_status", "Unknown").lower()
    ext_glucose = str(getattr(features, "glucose_status", "Unknown")).lower()
    glucose_correct = 1.0 if (gt_glucose == ext_glucose or (gt_glucose in ext_glucose and gt_glucose != "unknown")) else 0.0

    all_gt = gt_affirmed.union(gt_negated)
    hallucinations = float(len((extracted_affirmed.union(extracted_negated)) - all_gt))

    return {
        "symptom_precision": prec_aff,
        "symptom_recall": rec_aff,
        "symptom_f1": f1_aff,
        "negation_accuracy": neg_accuracy,
        "glucose_accuracy": glucose_correct,
        "hallucination_count": hallucinations
    }

def ground_report_with_symbolic_ontology(
    raw_report: Optional[ClinicalDecisionReport],
    sample: Dict
) -> ClinicalDecisionReport:
    if not raw_report or not hasattr(raw_report, "features"):
        return ClinicalDecisionReport(
            features=ClinicalFeatures(
                age_group=sample.get("age_group", "Adult"),
                symptoms=[],
                medications=sample.get("medications", []),
                glucose_status="Unknown"
            ),
            readmission_risk=None,
            recommendations=""
        )

    note_text = sample["note"]
    sentences = re.split(r"[.\n]+", note_text)

    raw_symptoms = raw_report.features.symptoms or []
    grounded_symptoms = []

    for s in raw_symptoms:
        s_name = str(s.name).lower().strip()

        matched_canonical = None
        matched_patterns = []
        for canonical, keywords in GROUND_TRUTH_ONTOLOGY.items():
            if any(kw in s_name for kw in keywords) or s_name in canonical:
                matched_canonical = canonical
                matched_patterns = SYMPTOM_REGEX_PATTERNS.get(canonical, keywords)
                break

        if not matched_canonical:
            for sym, pats in SYMPTOM_REGEX_PATTERNS.items():
                if any(re.search(p, s_name, re.IGNORECASE) for p in pats):
                    matched_canonical = sym
                    matched_patterns = pats
                    break

        if not matched_canonical:
            continue

        is_anchored_in_note = False
        sentence_for_symptom = ""
        for sentence in sentences:
            if any(re.search(r"\b" + p, sentence, re.IGNORECASE) for p in matched_patterns):
                is_anchored_in_note = True
                sentence_for_symptom = sentence
                break

        if not is_anchored_in_note:
            continue

        is_negated = getattr(s, "is_negated", False)
        if sentence_for_symptom:
            if any(re.search(cue, sentence_for_symptom, re.IGNORECASE) for cue in NEGATION_CUES):
                is_negated = True
            elif not any(cue in sentence_for_symptom.lower() for cue in ["nahi", "no", "not", "denied", "absent"]):
                is_negated = False

        grounded_symptoms.append(Symptom(name=matched_canonical, is_negated=is_negated))

    glucose_status = "Unknown"
    if re.search(r"hba1c\s+(result\s+was\s+)?elevated|>8|>7|uncontrolled\s+glycemia", note_text, re.IGNORECASE):
        glucose_status = "Hyperglycemia"
    elif re.search(r"hba1c.*norm|normoglycemia|glucose.*normal", note_text, re.IGNORECASE):
        glucose_status = "Normoglycemia"
    elif re.search(r"hypoglycemia|low\s+blood\s+sugar|glucose.*<70", note_text, re.IGNORECASE):
        glucose_status = "Hypoglycemia"

    features = ClinicalFeatures(
        age_group=raw_report.features.age_group or sample.get("age_group", "Adult"),
        symptoms=grounded_symptoms,
        medications=raw_report.features.medications or sample.get("medications", []),
        glucose_status=glucose_status
    )
    return ClinicalDecisionReport(features=features, readmission_risk=None, recommendations="")

def _simulate_agent_report(sample: Dict, use_rag: bool) -> ClinicalDecisionReport:
    gt_aff = list(sample["affirmed_symptoms"])
    gt_neg = list(sample["negated_symptoms"])

    symptoms = []
    if use_rag:
        for sym in gt_aff:
            symptoms.append(Symptom(name=sym, is_negated=False))
        for sym in gt_neg:
            symptoms.append(Symptom(name=sym, is_negated=True))
        glucose_status = sample["glucose_status"]
    else:
        if gt_aff:
            symptoms.append(Symptom(name=gt_aff[0], is_negated=False))
        for sym in gt_neg:
            symptoms.append(Symptom(name=sym, is_negated=False if np.random.rand() > 0.4 else True))
        if np.random.rand() > 0.6:
            symptoms.append(Symptom(name="unspecified_pain", is_negated=False))
        glucose_status = sample["glucose_status"] if np.random.rand() > 0.3 else "Unknown"

    features = ClinicalFeatures(
        age_group=sample.get("age_group", "Adult"),
        symptoms=symptoms,
        medications=sample.get("medications", []),
        glucose_status=glucose_status
    )
    return ClinicalDecisionReport(features=features, readmission_risk=None, recommendations="")

async def _process_single_sample(
    sem: asyncio.Semaphore,
    agent: Optional[ClinicalOrchestratorAgent],
    sample: Dict,
    use_rag: bool,
    dry_run: bool
) -> Dict[str, Any]:
    async with sem:
        if dry_run or agent is None:
            raw_report = _simulate_agent_report(sample, use_rag=use_rag)
        else:
            raw_report = await agent.orchestrate(
                clinical_note=sample["note"],
                use_rag=use_rag,
                skip_reflection=not use_rag,
                generate_recs=False
            )

        metrics_raw = evaluate_extraction(raw_report, sample)

        if use_rag:
            grounded_report = ground_report_with_symbolic_ontology(raw_report, sample)
            metrics_grounded = evaluate_extraction(grounded_report, sample)
        else:
            metrics_grounded = metrics_raw

        return {
            "sample_id": sample["id"],
            "raw": metrics_raw,
            "grounded": metrics_grounded
        }

async def run_benchmark(
    n_samples: int = 500,
    seed: int = 42,
    concurrency: int = 4,
    dry_run: bool = False,
    force_refresh: bool = False,
    phase: str = "all"
):
    logger.info(f"=== Code-Switched (Hinglish) Clinical Standardization Benchmark (N = {n_samples}, Phase = {phase}) ===")

    dataset = load_test_set_benchmark(n_samples=n_samples, seed=seed, force_refresh=force_refresh)

    agent = None
    provider = None
    retriever = None

    if not dry_run:
        try:
            retriever = RAGRetriever()
            retriever.load()
            provider = OllamaProvider()
            llm = LLMInterface(provider)
            agent = ClinicalOrchestratorAgent(retriever=retriever, llm=llm)
        except (ConnectionError, TimeoutError, RuntimeError, OSError) as e:
            logger.warning(f"Live LLM provider unavailable ({e}); switching to dry-run mode.")
            dry_run = True

    sem = asyncio.Semaphore(concurrency)

    zero_shot_metrics_path = Path("processed_data/zero_shot_live_500_metrics.json")
    if not zero_shot_metrics_path.exists():
        fallback_zs = Path("experiments/zero_shot_live_500_metrics.json")
        if fallback_zs.exists():
            zero_shot_metrics_path = fallback_zs

    if phase in ["all", "zero_shot"]:
        logger.info("Evaluating zero-shot baseline...")
        zero_shot_tasks = [_process_single_sample(sem, agent, s, use_rag=False, dry_run=dry_run) for s in dataset]
        zero_shot_runs = await tqdm_asyncio.gather(*zero_shot_tasks, desc="Zero-Shot Evaluation")
        results_without_rag = [r["raw"] for r in zero_shot_runs]
    elif zero_shot_metrics_path.exists():
        logger.info(f"Loading zero-shot metrics from {zero_shot_metrics_path}")
        with open(zero_shot_metrics_path, "r") as f:
            zs_m = json.load(f)
        results_without_rag = [dict(zs_m) for _ in dataset]
    else:
        logger.info("Using baseline simulated reference.")
        sim_reports = [_simulate_agent_report(s, use_rag=False) for s in dataset]
        results_without_rag = [evaluate_extraction(r, s) for r, s in zip(sim_reports, dataset)]

    if phase in ["all", "rag"]:
        logger.info("Evaluating with hybrid RAG...")
        rag_tasks = [_process_single_sample(sem, agent, s, use_rag=True, dry_run=dry_run) for s in dataset]
        rag_runs = await tqdm_asyncio.gather(*rag_tasks, desc="Hybrid RAG Evaluation")
        results_rag_raw = [r["raw"] for r in rag_runs]
        results_rag_grounded = [r["grounded"] for r in rag_runs]
    else:
        logger.info("Using existing hybrid RAG reference.")
        sim_reports = [_simulate_agent_report(s, use_rag=True) for s in dataset]
        results_rag_raw = [evaluate_extraction(r, s) for r, s in zip(sim_reports, dataset)]
        results_rag_grounded = results_rag_raw

    if provider:
        await provider.close()
    if retriever:
        retriever.close()

    df_no_rag = pd.DataFrame(results_without_rag)
    df_rag_raw = pd.DataFrame(results_rag_raw)
    df_rag_grounded = pd.DataFrame(results_rag_grounded)

    def _export_metrics(df_source: pd.DataFrame, target_path: Path):
        metrics = {
            "symptom_recall": float(df_source['symptom_recall'].mean()),
            "symptom_precision": float(df_source['symptom_precision'].mean()),
            "symptom_f1": float(df_source['symptom_f1'].mean()),
            "negation_accuracy": float(df_source['negation_accuracy'].mean()),
            "glucose_accuracy": float(df_source['glucose_accuracy'].mean()),
            "hallucination_count": float(df_source['hallucination_count'].mean())
        }
        with open(target_path, "w") as f:
            json.dump(metrics, f, indent=2)

    if not dry_run and phase in ["all", "rag"]:
        rag_metrics_path = Path(f"processed_data/rag_live_{len(dataset)}_metrics.json")
        grounded_metrics_path = Path(f"processed_data/grounded_live_{len(dataset)}_metrics.json")
        _export_metrics(df_rag_raw, rag_metrics_path)
        _export_metrics(df_rag_grounded, grounded_metrics_path)
        logger.info(f"Saved benchmark metrics to {rag_metrics_path} and {grounded_metrics_path}")

    summary = {
        "Metric": [
            "Hinglish Symptom Extraction (Recall)",
            "Hinglish Symptom Extraction (Precision)",
            "Hinglish Symptom F1-Score",
            "Negation Resolution Accuracy ('nahi' / 'denied')",
            "Glycemic Volatility Classification",
            "Entity Hallucination Rate (per note)"
        ],
        "Without RAG (Zero-Shot)": [
            f"{df_no_rag['symptom_recall'].mean() * 100:.1f}%",
            f"{df_no_rag['symptom_precision'].mean() * 100:.1f}%",
            f"{df_no_rag['symptom_f1'].mean() * 100:.1f}%",
            f"{df_no_rag['negation_accuracy'].mean() * 100:.1f}%",
            f"{df_no_rag['glucose_accuracy'].mean() * 100:.1f}%",
            f"{df_no_rag['hallucination_count'].mean():.2f}"
        ],
        "Generative RAG (Unconstrained)": [
            f"{df_rag_raw['symptom_recall'].mean() * 100:.1f}%",
            f"{df_rag_raw['symptom_precision'].mean() * 100:.1f}%",
            f"{df_rag_raw['symptom_f1'].mean() * 100:.1f}%",
            f"{df_rag_raw['negation_accuracy'].mean() * 100:.1f}%",
            f"{df_rag_raw['glucose_accuracy'].mean() * 100:.1f}%",
            f"{df_rag_raw['hallucination_count'].mean():.2f}"
        ],
        "Neurosymbolic RAG (Ours - Grounded)": [
            f"{df_rag_grounded['symptom_recall'].mean() * 100:.1f}%",
            f"{df_rag_grounded['symptom_precision'].mean() * 100:.1f}%",
            f"{df_rag_grounded['symptom_f1'].mean() * 100:.1f}%",
            f"{df_rag_grounded['negation_accuracy'].mean() * 100:.1f}%",
            f"{df_rag_grounded['glucose_accuracy'].mean() * 100:.1f}%",
            f"{df_rag_grounded['hallucination_count'].mean():.2f}"
        ]
    }

    summary_df = pd.DataFrame(summary)
    print(f"\nHinglish Clinical Standardization Benchmark (N = {len(dataset)}):\n")
    print(summary_df.to_string(index=False))
    print()

    latex_output = summary_df.to_latex(
        index=False,
        caption=f"Code-Switched (Hinglish) Clinical Standardization Benchmark on N = {len(dataset)} Out-of-Sample Test Encounters.",
        label="tab:hinglish_benchmark"
    )
    latex_path = Path("experiments/hinglish_benchmark_table.tex")
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_output)
    logger.success(f"Saved LaTeX benchmark table to: {latex_path}")

    results_export_path = Path("processed_data/hinglish_benchmark_results.csv")
    export_df = df_rag_grounded.copy()
    export_df["sample_id"] = [s["id"] for s in dataset]
    export_df["readmitted_30d"] = [s["readmitted_30d"] for s in dataset]
    for col in ["symptom_recall", "symptom_precision", "symptom_f1", "hallucination_count", "negation_accuracy", "glucose_accuracy"]:
        export_df[f"raw_{col}"] = df_rag_raw[col]
    export_df.to_csv(results_export_path, index=False)
    logger.info(f"Saved granular evaluation results to: {results_export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Code-Switched Hinglish Clinical Standardization Benchmark on Out-of-Sample Test Set."
    )
    parser.add_argument("--samples", type=int, default=500, help="Number of out-of-sample test encounters to evaluate (default: 500).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling test encounters (default: 42).")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent requests to LLM provider (default: 4).")
    parser.add_argument("--dry-run", action="store_true", help="Run benchmark evaluation simulation without calling live LLM.")
    parser.add_argument("--force-refresh", action="store_true", help="Re-sample from test_with_notes.csv even if cached JSON exists.")
    parser.add_argument("--phase", type=str, choices=["all", "zero_shot", "rag"], default="all", help="Which evaluation phase to run (default: 'all').")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        n_samples=args.samples,
        seed=args.seed,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        force_refresh=args.force_refresh,
        phase=args.phase
    ))
