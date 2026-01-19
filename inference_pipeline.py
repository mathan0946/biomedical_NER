"""
Complete Biomedical NER + RE Pipeline
Uses trained BioBERT models to extract entities and relationships from text.
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from peft import PeftModel
from itertools import combinations


# ============================================================================
# Configuration
# ============================================================================

NER_MODEL_PATH = "models/biobert-ner-lora"
RE_MODEL_PATH = "models/biobert-re-lora"
BASE_MODEL = "dmis-lab/biobert-base-cased-v1.2"

ENTITY_COLORS = {
    "GeneOrGeneProduct": "🔴",
    "DiseaseOrPhenotypicFeature": "🟢", 
    "ChemicalEntity": "🔵",
    "SequenceVariant": "🟢",
    "OrganismTaxon": "🟣",
    "CellLine": "🟡"
}


# ============================================================================
# NER Pipeline
# ============================================================================

class NERPipeline:
    """Named Entity Recognition pipeline."""
    
    def __init__(self, model_path: str = NER_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        self._load_model()
        
    def _load_model(self):
        """Load NER model and tokenizer."""
        print(f"Loading NER model from {self.model_path}...")
        
        # Load label mappings
        with open(self.model_path / "id2label.json", "r") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        with open(self.model_path / "label2id.json", "r") as f:
            self.label2id = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        base_model = AutoModelForTokenClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"NER model loaded on {self.device}")
    
    def predict(self, text: str) -> list:
        """Extract entities from text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        # Extract entities
        entities = []
        current_entity = None
        
        for idx, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
            if offset == [0, 0]:  # Skip special tokens
                continue
            
            label = self.id2label[pred_id]
            start, end = offset
            
            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    "type": entity_type,
                    "start": start,
                    "end": end,
                }
            
            elif label.startswith("I-") and current_entity:
                # Continue entity if same type
                entity_type = label[2:]
                if entity_type == current_entity["type"]:
                    current_entity["end"] = end
                else:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                    current_entity = None
            
            else:
                # O tag - save current entity if exists
                if current_entity:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity
        if current_entity:
            current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
            entities.append(current_entity)
        
        return [e for e in entities if e.get("text", "").strip()]


# ============================================================================
# Relation Extraction Pipeline
# ============================================================================

class REPipeline:
    """Relation Extraction pipeline."""
    
    def __init__(self, model_path: str = RE_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        self._load_model()
    
    def _load_model(self):
        """Load RE model and tokenizer."""
        print(f"Loading RE model from {self.model_path}...")
        
        # Load label mappings
        with open(self.model_path / "id2label.json", "r") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        with open(self.model_path / "label2id.json", "r") as f:
            self.label2id = json.load(f)
        
        # Load tokenizer with special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"RE model loaded on {self.device}")
    
    def predict(self, text: str, entity1: dict, entity2: dict) -> tuple:
        """Predict relationship between two entities."""
        # Sort entities by position
        if entity1["start"] < entity2["start"]:
            first, second = entity1, entity2
            first_markers = ("[E1]", "[/E1]")
            second_markers = ("[E2]", "[/E2]")
        else:
            first, second = entity2, entity1
            first_markers = ("[E2]", "[/E2]")
            second_markers = ("[E1]", "[/E1]")
        
        # Create marked text
        marked_text = (
            text[:first["start"]] +
            first_markers[0] + text[first["start"]:first["end"]] + first_markers[1] +
            text[first["end"]:second["start"]] +
            second_markers[0] + text[second["start"]:second["end"]] + second_markers[1] +
            text[second["end"]:]
        )
        
        # Tokenize
        inputs = self.tokenizer(
            marked_text,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
        
        relation = self.id2label[pred_id]
        return relation, confidence


# ============================================================================
# Complete Pipeline
# ============================================================================

def run_complete_pipeline(text: str, ner_pipeline: NERPipeline, re_pipeline: REPipeline, 
                          min_confidence: float = 0.5, max_pairs: int = 10):
    """Run complete NER + RE pipeline on text."""
    print("\n" + "="*80)
    print("BIOMEDICAL NER + RE PIPELINE")
    print("="*80)
    
    # Step 1: Extract entities
    print("\n[1/2] Extracting entities...")
    entities = ner_pipeline.predict(text)
    print(f"Found {len(entities)} entities:")
    
    for entity in entities:
        icon = ENTITY_COLORS.get(entity['type'], '⚪')
        print(f"  {icon} {entity['text']:<30} ({entity['type']})")
    
    # Step 2: Extract relations between entity pairs
    print(f"\n[2/2] Extracting relations...")
    relations = []
    
    if len(entities) >= 2:
        entity_pairs = list(combinations(entities, 2))[:max_pairs]
        
        for e1, e2 in entity_pairs:
            relation, confidence = re_pipeline.predict(text, e1, e2)
            
            if relation != "NO_RELATION" and confidence >= min_confidence:
                relations.append({
                    "entity1": e1["text"],
                    "entity1_type": e1["type"],
                    "entity2": e2["text"],
                    "entity2_type": e2["type"],
                    "relation": relation,
                    "confidence": confidence
                })
    
    print(f"Found {len(relations)} relations:")
    for rel in relations:
        print(f"  • {rel['entity1']} --[{rel['relation']}]--> {rel['entity2']} "
              f"(confidence: {rel['confidence']:.2%})")
    
    return {
        "text": text,
        "entities": entities,
        "relations": relations
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example biomedical text
    example_text = """Metformin is a first-line drug for type 2 diabetes mellitus. 
    It works by activating AMPK and inhibiting hepatic glucose production. 
    The OCT1 transporter mediates its uptake into hepatocytes."""
    
    # Load models
    print("\n🔧 Loading models...")
    ner_pipeline = NERPipeline()
    re_pipeline = REPipeline()
    
    # Process text
    print(f"\nInput text:\n{example_text.strip()}\n")
    
    result = run_complete_pipeline(
        text=example_text.strip(),
        ner_pipeline=ner_pipeline,
        re_pipeline=re_pipeline,
        min_confidence=0.5,
        max_pairs=10
    )
    
    # Save result
    output_file = "pipeline_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
