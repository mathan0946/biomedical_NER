"""
Fine-tune BioBERT with LoRA for Named Entity Recognition (NER).
Uses the BIORED dataset prepared in IOB format.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class NERConfig:
    """Configuration for NER training."""
    # Model
    model_name: str = "dmis-lab/biobert-base-cased-v1.2"
    
    # Data paths
    data_dir: str = "data/ner"
    output_dir: str = "models/biobert-ner-lora"
    
    # LoRA configuration
    lora_r: int = 16  # Rank of LoRA matrices
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )
    
    # Training hyperparameters
    max_length: int = 512  # Full length for better results
    batch_size: int = 8    # RTX 3050 has 4GB VRAM
    learning_rate: float = 2e-4
    num_epochs: int = 5    # 5 epochs for good results
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Evaluation
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Data Loading
# ============================================================================

def load_labels(data_dir: str) -> Dict:
    """Load label mappings."""
    labels_file = Path(data_dir) / "labels.txt"
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    # Ensure 'O' is the first label (index 0)
    if 'O' in labels:
        labels.remove('O')
    labels = ['O'] + sorted(labels)
    
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return labels, label2id, id2label


def load_jsonl_data(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_dataset(data_dir: str, split: str) -> Dataset:
    """Create HuggingFace Dataset from JSONL file."""
    filepath = Path(data_dir) / f"{split}.jsonl"
    data = load_jsonl_data(filepath)
    
    return Dataset.from_list(data)


# ============================================================================
# Tokenization & Alignment
# ============================================================================

def tokenize_and_align_labels(
    examples: Dict,
    tokenizer,
    label2id: Dict,
    max_length: int = 512
) -> Dict:
    """
    Tokenize text and align NER labels with tokenized output.
    
    Handles subword tokenization by:
    - Assigning the label to the first token of each word
    - Using -100 for subsequent subword tokens (ignored in loss)
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True,
        return_tensors=None,
    )
    
    labels = []
    for i, label_list in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                if word_idx < len(label_list):
                    label = label_list[word_idx]
                    label_ids.append(label2id.get(label, label2id['O']))
                else:
                    label_ids.append(-100)
            else:
                # Subsequent subword tokens get -100
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(p, id2label: Dict):
    """Compute NER metrics using seqeval."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (-100) and convert to labels
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        pred_tags = []
        true_tags = []
        
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                pred_tags.append(id2label[pred_id])
                true_tags.append(id2label[label_id])
        
        true_predictions.append(pred_tags)
        true_labels.append(true_tags)
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


# ============================================================================
# Model Setup
# ============================================================================

def create_model_with_lora(config: NERConfig, num_labels: int, id2label: Dict, label2id: Dict):
    """Create BioBERT model with LoRA adapters."""
    
    print(f"\n📦 Loading model: {config.model_name}")
    
    # Load base model
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


# ============================================================================
# Training
# ============================================================================

def train_ner(config: NERConfig):
    """Main training function."""
    
    print("\n" + "="*60)
    print("  🧬 BioBERT + LoRA NER Training")
    print("="*60)
    
    # Load labels
    print("\n📋 Loading labels...")
    labels, label2id, id2label = load_labels(config.data_dir)
    print(f"   Found {len(labels)} labels: {labels}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = create_dataset(config.data_dir, "train")
    eval_dataset = create_dataset(config.data_dir, "dev")
    test_dataset = create_dataset(config.data_dir, "test")
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Dev:   {len(eval_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Tokenize datasets
    print("\n🔄 Tokenizing datasets...")
    tokenize_fn = lambda examples: tokenize_and_align_labels(
        examples, tokenizer, label2id, config.max_length
    )
    
    train_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing dev"
    )
    
    test_tokenized = test_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test"
    )
    
    # Create model with LoRA
    model = create_model_with_lora(config, len(labels), id2label, label2id)
    model.to(config.device)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    # Create compute_metrics function with id2label
    def compute_metrics_fn(p):
        return compute_metrics(p, id2label)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"   Device: {config.device}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   LoRA rank: {config.lora_r}")
    
    trainer.train()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    
    print(f"\n{'='*60}")
    print("  Test Results")
    print(f"{'='*60}")
    print(f"   Precision: {test_results['eval_precision']:.4f}")
    print(f"   Recall:    {test_results['eval_recall']:.4f}")
    print(f"   F1 Score:  {test_results['eval_f1']:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save label mappings
    with open(Path(config.output_dir) / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(Path(config.output_dir) / "id2label.json", 'w') as f:
        json.dump(id2label, f, indent=2)
    
    print("\n✅ NER training complete!")
    
    return trainer, test_results


# ============================================================================
# Inference
# ============================================================================

def load_ner_model(model_dir: str, device: str = None):
    """Load trained NER model for inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load label mappings
    with open(Path(model_dir) / "label2id.json", 'r') as f:
        label2id = json.load(f)
    with open(Path(model_dir) / "id2label.json", 'r') as f:
        id2label = json.load(f)
    # Convert string keys to int for id2label
    id2label = {int(k): v for k, v in id2label.items()}
    
    # Load base model
    base_model_name = "dmis-lab/biobert-base-cased-v1.2"
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.merge_and_unload()  # Merge LoRA weights for faster inference
    model.to(device)
    model.eval()
    
    return model, tokenizer, id2label


def predict_entities(text: str, model, tokenizer, id2label: Dict, device: str = None):
    """Predict entities in text."""
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
    
    # Extract entities
    entities = []
    current_entity = None
    
    for idx, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
        if offset == [0, 0]:  # Skip special tokens
            continue
        
        label = id2label[pred_id]
        
        if label.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append(current_entity)
            
            # Start new entity
            entity_type = label[2:]
            current_entity = {
                "text": text[offset[0]:offset[1]],
                "type": entity_type,
                "start": offset[0],
                "end": offset[1],
            }
        
        elif label.startswith("I-") and current_entity:
            # Continue entity
            entity_type = label[2:]
            if entity_type == current_entity["type"]:
                current_entity["text"] = text[current_entity["start"]:offset[1]]
                current_entity["end"] = offset[1]
            else:
                # Type mismatch, save and start new
                entities.append(current_entity)
                current_entity = None
        
        else:
            # O tag or mismatch
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = NERConfig()
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer, results = train_ner(config)
