"""
Fine-tune BioBERT with QLoRA for Relation Extraction (RE).
Optimized for RTX 3050 4GB VRAM.

Uses:
  - QLoRA (4-bit quantization + LoRA adapters)
  - Focal Loss with class weighting for imbalanced data
  - Smart dataset balancing (cap majority, oversample minority)
"""

import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# ============================================================================
# Focal Loss for Hard Examples
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss to focus on hard examples and handle class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focus parameter
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none',
            weight=self.alpha.to(inputs.device) if self.alpha is not None else None
        )
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma * ce_loss).mean()
        return focal_loss


class FocalTrainer(Trainer):
    """Custom trainer with focal loss and class weighting."""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class REConfig:
    """Configuration for RE training with QLoRA and Focal Loss."""
    model_name: str = "dmis-lab/biobert-base-cased-v1.2"
    data_dir: str = "data/re"
    output_dir: str = "models/biobert-re-lora"

    # QLoRA parameters
    lora_r: int = 24
    lora_alpha: int = 48
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )

    # Training parameters
    max_length: int = 240
    batch_size: int = 6
    gradient_accumulation_steps: int = 4  # Effective batch = 24
    learning_rate: float = 2.5e-4
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Dataset balancing
    max_samples_per_class: int = 6000  # Cap majority classes
    min_samples_per_class: int = 2000  # Oversample minority classes

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    early_stopping_patience: int = 5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Data helpers
# ============================================================================

def load_labels(data_dir: str):
    with open(Path(data_dir) / "labels.txt", "r", encoding="utf-8") as f:
        labels = sorted([l.strip() for l in f if l.strip()])
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def load_jsonl(filepath: str):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def balance_dataset(data: List[Dict], min_per_class: int, max_per_class: int, seed: int = 42):
    """Balance dataset: cap majority classes, oversample minority classes."""
    random.seed(seed)
    
    by_label = {}
    for sample in data:
        by_label.setdefault(sample["label"], []).append(sample)
    
    balanced = []
    for label, samples in by_label.items():
        if len(samples) > max_per_class:
            # Undersample large classes
            sampled = random.sample(samples, max_per_class)
            balanced.extend(sampled)
            print(f"  ✓ {label}: {len(samples)} → {max_per_class} (capped)")
        elif len(samples) < min_per_class:
            # Oversample small classes
            oversampled = random.choices(samples, k=min_per_class)
            balanced.extend(oversampled)
            print(f"  ✓ {label}: {len(samples)} → {min_per_class} (oversampled)")
        else:
            balanced.extend(samples)
            print(f"  ✓ {label}: {len(samples)} (kept all)")
    
    random.shuffle(balanced)
    return balanced


def compute_class_weights(data: List[Dict], labels: List[str]) -> torch.Tensor:
    """Compute inverse frequency class weights with smoothing."""
    label_counts = Counter(d["label"] for d in data)
    total = len(data)
    num_classes = len(labels)
    
    weights = []
    for label in labels:
        count = label_counts.get(label, 1)
        # Inverse frequency with sqrt smoothing
        weight = np.sqrt(total / (num_classes * count))
        weights.append(min(weight, 5.0))  # Cap at 5x
    
    return torch.tensor(weights, dtype=torch.float32)


def create_dataset(data_dir: str, split: str, min_per_class: int = None, max_per_class: int = None) -> Dataset:
    """Load dataset and balance if training split."""
    data = load_jsonl(Path(data_dir) / f"{split}.jsonl")
    
    if split == "train" and min_per_class and max_per_class:
        data = balance_dataset(data, min_per_class, max_per_class)
    
    return Dataset.from_list(data)


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_fn(examples, tokenizer, label2id, max_length):
    tok = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    tok["labels"] = [label2id[l] for l in examples["label"]]
    return tok


# ============================================================================
# Model
# ============================================================================

def build_model(config: REConfig, num_labels, id2label, label2id):
    """Load BioBERT in 4-bit and attach LoRA adapters."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(pred, id2label):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


# ============================================================================
# Training
# ============================================================================

def train_re(config: REConfig):

    print("\n" + "=" * 60)
    print("  BioBERT + QLoRA + Focal Loss")
    print("=" * 60)

    labels, label2id, id2label = load_labels(config.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    )

    # Load datasets with balanced training data
    print("\nBalancing training dataset...")
    train_ds = create_dataset(
        config.data_dir, "train",
        min_per_class=config.min_samples_per_class,
        max_per_class=config.max_samples_per_class
    )
    eval_ds = create_dataset(config.data_dir, "dev")
    test_ds = create_dataset(config.data_dir, "test")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Dev:   {len(eval_ds)} | Test: {len(test_ds)}")

    # Compute class weights for focal loss
    orig_train = load_jsonl(Path(config.data_dir) / "train.jsonl")
    class_weights = compute_class_weights(orig_train, labels)
    print(f"\nClass weights: {dict(zip(labels, [f'{w:.2f}' for w in class_weights]))}")

    # Tokenize
    cols = list(train_ds.column_names)
    map_fn = lambda ex: tokenize_fn(ex, tokenizer, label2id, config.max_length)

    train_tok = train_ds.map(map_fn, batched=True, remove_columns=cols, desc="Tok train")
    eval_tok  = eval_ds.map(map_fn, batched=True, remove_columns=cols, desc="Tok dev")
    test_tok  = test_ds.map(map_fn, batched=True, remove_columns=cols, desc="Tok test")

    # Build model with QLoRA
    model = build_model(config, len(labels), id2label, label2id)
    model.resize_token_embeddings(len(tokenizer))

    # Dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    steps_per_epoch = len(train_tok) // (config.batch_size * config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_epochs
    print(f"\nTraining plan:")
    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps}")
    print(f"  Estimated time: ~{total_steps * 1.5 / 60:.0f} min")

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        fp16=True,
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.05,
    )

    def metrics_fn(p):
        return compute_metrics(p, id2label)

    # Use FocalTrainer with class weights
    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    print("\n🚀 Starting training with Focal Loss...")
    trainer.train()

    # Test evaluation
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_tok)

    print(f"\n{'=' * 60}")
    print("  Test Results")
    print(f"{'=' * 60}")
    print(f"   Accuracy:      {test_results['eval_accuracy']:.4f}")
    print(f"   Precision:     {test_results['eval_precision']:.4f}")
    print(f"   Recall:        {test_results['eval_recall']:.4f}")
    print(f"   F1 (weighted): {test_results['eval_f1']:.4f}")
    print(f"   F1 (macro):    {test_results['eval_f1_macro']:.4f}")

    # Detailed report
    predictions = trainer.predict(test_tok)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    pred_names = [id2label[p] for p in pred_labels]
    true_names = [id2label[l] for l in true_labels]
    print("\nClassification Report:")
    print(classification_report(true_names, pred_names, zero_division=0))

    # Save
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    with open(Path(config.output_dir) / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open(Path(config.output_dir) / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print("\n✅ Training complete! Model saved to", config.output_dir)
    return trainer, test_results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = REConfig()
    train_re(config)
