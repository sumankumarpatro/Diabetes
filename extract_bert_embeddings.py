import argparse
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from config import config

class ClinicalNotesDataset(Dataset):
    def __init__(self, notes: list[str]):
        self.notes = notes

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        return str(self.notes[idx])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def extract_embeddings_for_dataset(
    csv_path: Path,
    output_npy_path: Path,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int = 128,
    max_length: int = 256
) -> None:
    if not csv_path.exists():
        logger.error(f"Source file not found at: {csv_path}")
        return

    logger.info(f"Loading clinical notes from: {csv_path}")
    df = pd.read_csv(csv_path)

    if 'clinical_note' not in df.columns:
        logger.error(f"'clinical_note' column missing in {csv_path}")
        return

    notes = df['clinical_note'].fillna("").astype(str).tolist()
    total_notes = len(notes)
    logger.info(f"Loaded {total_notes} notes. Compiling batches (batch_size={batch_size}, max_len={max_length})...")

    dataset = ClinicalNotesDataset(notes)
    
    def collate_fn(batch):
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding {csv_path.name} via MPS"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pooling(outputs, attention_mask)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

    dense_matrix = np.vstack(all_embeddings)
    logger.info(f"Dense embedding shape: {dense_matrix.shape} (Expected: ({total_notes}, {config.BERT_EMBEDDING_DIM}))")

    if dense_matrix.shape[0] != total_notes:
        raise ValueError(f"Shape mismatch! Expected {total_notes} rows, got {dense_matrix.shape[0]}")

    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_npy_path), dense_matrix)
    logger.success(f"Successfully saved dense embeddings to: {output_npy_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract Bio_ClinicalBERT dense embeddings.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for MPS GPU inference.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence token length.")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Initializing Bio_ClinicalBERT embedding pipeline on device: [{device.upper()}]")

    model_name = config.BERT_MODEL_NAME
    logger.info(f"Loading pretrained tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    logger.info("--- train split ---")
    extract_embeddings_for_dataset(
        csv_path=config.TRAIN_WITH_NOTES_PATH,
        output_npy_path=config.TRAIN_BERT_EMBEDDINGS_PATH,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    logger.info("--- test split ---")
    extract_embeddings_for_dataset(
        csv_path=config.TEST_WITH_NOTES_PATH,
        output_npy_path=config.TEST_BERT_EMBEDDINGS_PATH,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    logger.success("done, embeddings saved for both splits")

if __name__ == "__main__":
    main()