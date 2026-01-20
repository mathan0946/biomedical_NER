"""
Fine-tune BioBERT with LoRA for Relation Extraction (RE).
Uses the BIORED dataset prepared with entity-marked text.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class REConfig:
    """Configuration for RE training."""
    # Model
    model_name: str = "dmis-lab/biobert-base-cased-v1.2"
    
    # Data paths
    data_dir: str = "data/re"
    output_dir: str = "models/biobert-re-lora"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value"]
    )
    
    # Training hyperparameters
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # Evaluation
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    
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
    
    labels = sorted(labels)
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
# Tokenization
# ============================================================================

def tokenize_function(
    examples: Dict,
    tokenizer,
    label2id: Dict,
    max_length: int = 256
) -> Dict:
    """Tokenize text for sequence classification."""
    
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    
    # Convert labels to ids
    tokenized["labels"] = [
        label2id[label] for label in examples["label"]
    ]
    
    return tokenized


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(p, id2label: Dict):
    """Compute classification metrics."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    # Convert to label names for report
    pred_labels = [id2label[p] for p in predictions]
    true_labels = [id2label[l] for l in labels]
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


# ============================================================================
# Model Setup
# ============================================================================

def create_model_with_lora(config: REConfig, num_labels: int, id2label: Dict, label2id: Dict):
    """Create BioBERT model with LoRA adapters for sequence classification."""
    
    # Configure 8-bit quantization for memory efficiency
    quantization_config = None
    if config.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        quantization_config=quantization_config,
        device_map="auto" if config.use_8bit else None,
    )
    
    # Enable gradient checkpointing for memory savings
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
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

def train_re(config: REConfig):
    """Main training function for RE."""
    
    print("\n" + "="*60)
    print("  🔗 BioBERT + LoRA Relation Extraction Training")
    print("="*60)
    
    # Load labels
    labels, label2id, id2label = load_labels(config.data_dir)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add special tokens for entity markers
    special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Load datasets
    train_dataset = create_dataset(config.data_dir, "train")
    eval_dataset = create_dataset(config.data_dir, "dev")
    test_dataset = create_dataset(config.data_dir, "test")
    
    # Tokenize datasets
    tokenize_fn = lambda examples: tokenize_function(
        examples, tokenizer, label2id, config.max_length
    )
    
    columns_to_remove = ["doc_id", "text", "entity1", "entity1_type", 
                         "entity2", "entity2_type", "relation", "label"]
    
    train_tokenized = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in columns_to_remove if c in train_dataset.column_names],
        desc="Tokenizing train"
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in columns_to_remove if c in eval_dataset.column_names],
        desc="Tokenizing dev"
    )
    
    test_tokenized = test_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in columns_to_remove if c in test_dataset.column_names],
        desc="Tokenizing test"
    )
    
    # Create model with LoRA
    model = create_model_with_lora(config, len(labels), id2label, label2id)
    
    # Resize embeddings for new special tokens
    model.resize_token_embeddings(len(tokenizer))
    model.to(config.device)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )
    
    # Create compute_metrics function
    def compute_metrics_fn(p):
        return compute_metrics(p, id2label)
    
    # Create trainer
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
    trainer.train()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_tokenized)
    
    print(f"\n{'='*60}")
    print("  Test Results")
    print(f"{'='*60}")
    print(f"   Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"   Precision: {test_results['eval_precision']:.4f}")
    print(f"   Recall:    {test_results['eval_recall']:.4f}")
    print(f"   F1 (weighted): {test_results['eval_f1']:.4f}")
    print(f"   F1 (macro):    {test_results['eval_f1_macro']:.4f}")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    predictions = trainer.predict(test_tokenized)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    pred_names = [id2label[p] for p in pred_labels]
    true_names = [id2label[l] for l in true_labels]
    
    print(classification_report(true_names, pred_names))
    
    # Save model
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save label mappings
    with open(Path(config.output_dir) / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(Path(config.output_dir) / "id2label.json", 'w') as f:
        json.dump(id2label, f, indent=2)
    
    print("\n✅ Training complete! Model saved to", config.output_dir)
    
    return trainer, test_results


# ============================================================================
# Inference
# ============================================================================

def load_re_model(model_dir: str, device: str = None):
    """Load trained RE model for inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load label mappings
    with open(Path(model_dir) / "label2id.json", 'r') as f:
        label2id = json.load(f)
    with open(Path(model_dir) / "id2label.json", 'r') as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    
    # Load base model
    base_model_name = "dmis-lab/biobert-base-cased-v1.2"
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    # Resize for special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    
    return model, tokenizer, id2label


def predict_relation(
    text: str,
    entity1: str,
    entity2: str,
    model,
    tokenizer,
    id2label: Dict,
    device: str = None
):
    """Predict relation between two entities in text."""
    if device is None:
        device = next(model.parameters()).device
    
    # Mark entities in text
    marked_text = text.replace(entity1, f"[E1] {entity1} [/E1]", 1)
    marked_text = marked_text.replace(entity2, f"[E2] {entity2} [/E2]", 1)
    
    # Tokenize
    inputs = tokenizer(
        marked_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
    
    return {
        "relation": id2label[pred_id],
        "confidence": confidence,
        "all_probs": {id2label[i]: p.item() for i, p in enumerate(probs)}
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = REConfig()
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer, results = train_re(config)
