"""
Biomedical NER and Relation Extraction Pipeline
Uses trained BioBERT models for entity and relation extraction.
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
        
        # Load tokenizer (which includes the added special tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model with correct vocabulary size
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        # Resize embeddings to match the tokenizer vocabulary
        base_model.resize_token_embeddings(len(self.tokenizer))
        
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"RE model loaded on {self.device}")
    
    def predict(self, text: str, entities: list, debug: bool = False) -> list:
        """Extract relations between entities."""
        relations = []
        
        # Generate all entity pairs
        for e1, e2 in combinations(entities, 2):
            # Create marked text with entity markers
            marked_text = self._create_marked_text(text, e1, e2)
            
            # Tokenize
            inputs = self.tokenizer(
                marked_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=0)
                pred_id = torch.argmax(logits).item()
                confidence = probs[pred_id].item()
            
            relation_type = self.id2label[pred_id]
            
            # Debug: show all predictions
            if debug:
                print(f"\n  {e1['text']} <-> {e2['text']}")
                print(f"  Predicted: {relation_type} (conf: {confidence:.3f})")
                print(f"  All probabilities:")
                for label_id, prob in enumerate(probs.cpu().tolist()):
                    print(f"    {self.id2label[label_id]}: {prob:.3f}")
            
            # Keep non-NO_RELATION relations with reasonable confidence
            if relation_type != "NO_RELATION" and confidence > 0.5:
                relations.append({
                    "head": e1["text"],
                    "head_type": e1["type"],
                    "tail": e2["text"],
                    "tail_type": e2["type"],
                    "relation": relation_type,
                    "confidence": round(confidence, 3)
                })
        
        return relations
    
    def _create_marked_text(self, text: str, e1: dict, e2: dict) -> str:
        """Create text with entity markers for relation extraction.
        Uses [E1] and [E2] markers matching the training data format.
        """
        # Sort entities by position to insert markers correctly
        if e1["start"] < e2["start"]:
            first, second = e1, e2
            is_e1_first = True
        else:
            first, second = e2, e1
            is_e1_first = False
        
        # Insert markers using the same format as training data: [E1] entity [/E1]
        if is_e1_first:
            marked = (
                text[:first["start"]] +
                "[E1] " + first["text"] + " [/E1]" +
                text[first["end"]:second["start"]] +
                "[E2] " + second["text"] + " [/E2]" +
                text[second["end"]:]
            )
        else:
            marked = (
                text[:first["start"]] +
                "[E2] " + first["text"] + " [/E2]" +
                text[first["end"]:second["start"]] +
                "[E1] " + second["text"] + " [/E1]" +
                text[second["end"]:]
            )
        
        return marked


# ============================================================================
# Combined Pipeline
# ============================================================================

class BiomedicalPipeline:
    """Combined NER + RE pipeline."""
    
    def __init__(self):
        self.ner = NERPipeline()
        self.re = REPipeline()
    
    def process(self, text: str, debug: bool = False) -> dict:
        """Process text through NER and RE."""
        # Extract entities
        entities = self.ner.predict(text)
        
        # Extract relations
        relations = self.re.predict(text, entities, debug=debug) if len(entities) >= 2 else []
        
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
    example_text = """Metformin is used to treat type 2 diabetes mellitus. 
Studies show that AMPK activation by metformin reduces glucose production in the liver. 
The OCT1 transporter is responsible for metformin uptake."""
    
    # Load pipeline
    print("\n🔧 Loading Biomedical NER + RE Pipeline...")
    pipeline = BiomedicalPipeline()
    
    # Process text
    print(f"\nInput text:\n{example_text.strip()}\n")
    print("\n" + "="*80)
    print("BIOMEDICAL NER + RELATION EXTRACTION PIPELINE")
    print("="*80)
    
    print("\n📝 Processing with debug info...")
    result = pipeline.process(example_text.strip(), debug=True)
    
    # Display entities
    print(f"\n{'='*80}")
    print(f"✓ Found {len(result['entities'])} entities:")
    for entity in result['entities']:
        icon = ENTITY_COLORS.get(entity['type'], '⚪')
        print(f"  {icon} {entity['text']:<30} ({entity['type']})")
    
    # Display relations
    print(f"\n✓ Found {len(result['relations'])} relations:")
    for rel in result['relations']:
        print(f"  → {rel['head']} -- [{rel['relation']}] --> {rel['tail']} (confidence: {rel['confidence']})")
    
    # Save result
    output_file = "pipeline_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")
