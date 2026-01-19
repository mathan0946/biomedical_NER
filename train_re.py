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
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples and class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Move alpha to same device as inputs
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=alpha)
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedTrainer(Trainer):
    """Custom trainer with focal loss and class weights for imbalanced data."""
    
    def __init__(self, class_weights=None, use_focal_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.loss_fct = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            self.loss_fct = None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.use_focal_loss and self.loss_fct is not None:
            loss = self.loss_fct(logits, labels)
        elif self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


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
    
    # LoRA configuration (higher rank = more capacity)
    lora_r: int = 48  # Higher rank for better performance
    lora_alpha: int = 96  # 2x rank
    lora_dropout: float = 0.05  # Lower dropout
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )
    
    # Training hyperparameters (optimized for 4GB GPU)
    max_length: int = 384  # Longer context for better relations
    batch_size: int = 3    # Reduced for longer sequences
    learning_rate: float = 1.5e-4  # Lower for stability with more data
    num_epochs: int = 8  # More epochs for convergence
    warmup_ratio: float = 0.15  # More warmup
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 10  # Effective batch = 30
    
    # 8-bit quantization for memory efficiency
    use_8bit: bool = False  # Disabled due to Windows compatibility
    
    # Class weighting for imbalanced data
    use_class_weights: bool = True
    
    # Data sampling (to handle class imbalance)
    max_samples_per_class: int = None  # Use all data for best performance
    
    # Evaluation
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    
    # Early stopping
    early_stopping_patience: int = 5  # More patience for convergence
    
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


def balance_dataset(data: List[Dict], max_per_class: int = None) -> List[Dict]:
    """Balance dataset by limiting samples per class."""
    if max_per_class is None:
        return data
    
    # Group by label
    by_label = {}
    for sample in data:
        label = sample['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(sample)
    
    # Sample from each class
    balanced = []
    for label, samples in by_label.items():
        if len(samples) > max_per_class:
            import random
            random.seed(42)
            samples = random.sample(samples, max_per_class)
        balanced.extend(samples)
    
    return balanced


def create_dataset(data_dir: str, split: str, max_per_class: int = None) -> Dataset:
    """Create HuggingFace Dataset from JSONL file."""
    filepath = Path(data_dir) / f"{split}.jsonl"
    data = load_jsonl_data(filepath)
    
    if max_per_class and split == "train":
        data = balance_dataset(data, max_per_class)
    
    return Dataset.from_list(data)


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_function(
    examples: Dict,
    tokenizer,
    label2id: Dict,
    max_length: int = 512,
    add_entity_types: bool = True
) -> Dict:
    """Tokenize text for sequence classification with optional entity type info."""
    
    texts = examples["text"]
    
    # Optionally enhance with entity type information
    if add_entity_types and "entity1_type" in examples and "entity2_type" in examples:
        enhanced_texts = []
        for text, e1_type, e2_type in zip(texts, examples["entity1_type"], examples["entity2_type"]):
            # Add entity type hints in the text
            text_with_types = text.replace("[E1]", f"[E1:{e1_type}]").replace("[E2]", f"[E2:{e2_type}]")
            enhanced_texts.append(text_with_types)
        texts = enhanced_texts
    
    tokenized = tokenizer(
        texts,
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
    
    print(f"\n📦 Loading model: {config.model_name}")
    
    # Configure 8-bit quantization for memory efficiency
    quantization_config = None
    if config.use_8bit:
        print("   Using 8-bit quantization for memory efficiency")
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
        print("   Gradient checkpointing enabled")
    
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
    print("\n📋 Loading labels...")
    labels, label2id, id2label = load_labels(config.data_dir)
    print(f"   Found {len(labels)} labels: {labels}")
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add special tokens for entity markers (with and without types)
    entity_types = ["Gene", "Disease", "Chemical", "Variant", "CellLine", "Organism"]
    special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    # Add type-specific markers
    for etype in entity_types:
        special_tokens.extend([f"[E1:{etype}]", f"[E2:{etype}]"])
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_dataset = create_dataset(config.data_dir, "train", config.max_samples_per_class)
    eval_dataset = create_dataset(config.data_dir, "dev")
    test_dataset = create_dataset(config.data_dir, "test")
    
    print(f"   Train: {len(train_dataset)} samples (balanced)")
    print(f"   Dev:   {len(eval_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Show class distribution
    train_labels = [s['label'] for s in train_dataset]
    label_counts = Counter(train_labels)
    print("\n   Training class distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"     • {label}: {count}")
    
    # Compute class weights for imbalanced data
    class_weights = None
    if config.use_class_weights:
        print("\n⚖️  Computing class weights for imbalanced data...")
        total_samples = len(train_labels)
        num_classes = len(labels)
        weights = []
        for label in labels:
            count = label_counts.get(label, 1)
            # Inverse frequency weighting with smoothing
            weight = total_samples / (num_classes * count)
            weights.append(min(weight, 10.0))  # Cap at 10x to avoid extreme weights
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"   Class weights: {dict(zip(labels, [f'{w:.2f}' for w in weights]))}")
    
    # Tokenize datasets with entity type information
    print("\n🔄 Tokenizing datasets (with entity type hints)...")
    tokenize_fn = lambda examples: tokenize_function(
        examples, tokenizer, label2id, config.max_length, add_entity_types=True
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
    
    # Only move to device if not using 8-bit (8-bit handles device automatically)
    if not config.use_8bit:
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
        per_device_eval_batch_size=config.batch_size * 2,  # Larger eval batch
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,  # Keep more checkpoints
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Avoid multiprocessing overhead on Windows
        dataloader_pin_memory=True,
        label_smoothing_factor=0.1,  # Label smoothing for regularization
        lr_scheduler_type="cosine",  # Cosine annealing for better convergence
    )
    
    # Create compute_metrics function
    def compute_metrics_fn(p):
        return compute_metrics(p, id2label)
    
    # Use WeightedTrainer with focal loss if class weights are enabled
    TrainerClass = WeightedTrainer if config.use_class_weights else Trainer
    
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights if config.use_class_weights else None,
        use_focal_loss=True if config.use_class_weights else False,  # Enable focal loss
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
    print(f"\n💾 Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Save label mappings
    with open(Path(config.output_dir) / "label2id.json", 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(Path(config.output_dir) / "id2label.json", 'w') as f:
        json.dump(id2label, f, indent=2)
    
    print("\n✅ RE training complete!")
    
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
