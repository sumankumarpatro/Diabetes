from loguru import logger
import pandas as pd
import numpy as np
import random
import os
from config import config

def generate_hinglish_note(row: pd.Series) -> str:
    """
    Generates a synthetic, multilingual (Hinglish) clinical note from a row of diabetes data.
    """
    # Templates for different styles of notes
    # Template 1: Very messy/informal
    # Template 2: Slightly more structured but still code-switched
    # Template 3: Short and direct
    
    age = row['age']
    time_in_hosp = row['time_in_hospital']
    num_labs = row['num_lab_procedures']
    num_meds = row['num_medications']
    diabetes_med = "diabetes medication" if row['diabetesMed'] == 'Yes' else "no specific diabetes meds"
    
    # Hinglish vocabulary/phrases
    hinglish_phrases = [
        "Patient ko bahut zyada sugar issues hain.",
        "Bukhar (fever) and weakness reported.",
        "Dard (pain) in the body.",
        "Medicine le rahe hain (taking medicine).",
        "Sugar level check kiya gaya (sugar level was checked).",
        "Patient ki condition stable hai.",
        "Patient ko bahut thakan (fatigue) mehsoos ho rahi hai."
    ]
    
    templates = [
        f"Patient age {age} presented with high sugar issues. {random.choice(hinglish_phrases)} Time in hospital: {time_in_hosp} days. {random.choice(hinglish_phrases)}",
        f"Clinical note: Age {age}. Patient is taking {diabetes_med}. {random.choice(hinglish_phrases)} Lab procedures done: {num_labs}. Number of medications: {num_meds}.",
        f"Patient {age} years old. {random.choice(hinglish_phrases)} Number of medications: {num_meds}. Hospital stay: {time_in_hosp} days. {random.choice(hinglish_phrases)}",
        f"Summary: {age} age patient. {random.choice(hinglish_phrases)} Lab tests: {num_labs}. Hospital stay: {time_in_hosp} days.",
        f"Age: {age}. {random.choice(hinglish_phrases)} Patient was in hospital for {time_in_hosp} days and took {num_meds} medications.",
        f"Note: {age} year old patient. {random.choice(hinglish_phrases)} Number of medications: {num_meds}. Diabetes medicine status: {diabetes_med}."
    ]
    
    return random.choice(templates)

def main():
    # Define the pairs of (input_path, output_path) to process
    tasks = [
        (config.TRAIN_DATA_PATH, config.TRAIN_WITH_NOTES_PATH),
        (config.TEST_DATA_PATH, config.TEST_WITH_NOTES_PATH)
    ]

    random.seed(42)

    for input_path, output_path in tasks:
        if not input_path.exists():
            logger.error(f"Input data not found: {input_path}")
            continue

        logger.info(f"Loading data from: {input_path}")
        df = pd.read_csv(input_path)

        logger.info(f"Generating Hinglish clinical notes for {input_path.name}...")
        # Apply the generation function to each row
        df['clinical_note'] = df.apply(generate_hinglish_note, axis=1)

        # Save the new dataset
        df.to_csv(output_path, index=False)
        logger.success(f"Successfully generated notes and saved to: {output_path}")
        logger.info(f"Sample note:\n{df['clinical_note'].iloc[0]}")

if __name__ == "__main__":
    main()
