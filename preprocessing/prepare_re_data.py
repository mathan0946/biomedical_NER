"""
Prepare Relation Extraction (RE) data from BIORED dataset.
Creates training samples with entity pairs and relation labels.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import random

random.seed(42)


def parse_pubtator_file(filepath: str) -> List[Dict]:
    """Parse PubTator format file and extract documents with annotations."""
    documents = []
    current_doc = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if current_doc:
                    documents.append(current_doc)
                    current_doc = None
                continue
            
            parts = line.split('|')
            
            # Title line
            if len(parts) == 3 and parts[1] == 't':
                current_doc = {
                    'id': parts[0],
                    'title': parts[2],
                    'abstract': '',
                    'entities': [],
                    'relations': []
                }
            # Abstract line
            elif len(parts) == 3 and parts[1] == 'a':
                if current_doc:
                    current_doc['abstract'] = parts[2]
            # Entity/Relation annotation
            elif '\t' in line:
                parts = line.split('\t')
                
                # Relation line (type is not a number)
                if len(parts) >= 4 and not parts[1].isdigit():
                    if current_doc:
                        current_doc['relations'].append({
                            'type': parts[1],
                            'arg1': parts[2],
                            'arg2': parts[3],
                            'novel': parts[4] if len(parts) > 4 else 'No'
                        })
                # Entity annotation
                elif len(parts) >= 5:
                    if current_doc:
                        current_doc['entities'].append({
                            'start': int(parts[1]),
                            'end': int(parts[2]),
                            'text': parts[3],
                            'type': parts[4],
                            'identifier': parts[5] if len(parts) > 5 else ''
                        })
    
    # Add last document
    if current_doc:
        documents.append(current_doc)
    
    return documents


def mark_entities_in_text(text: str, entity1: Dict, entity2: Dict) -> str:
    """
    Mark two entities in text with special tokens.
    Entity markers: [E1] entity1 [/E1] and [E2] entity2 [/E2]
    """
    # Sort entities by start position (process from end to avoid offset issues)
    entities = [(entity1, 'E1'), (entity2, 'E2')]
    entities.sort(key=lambda x: x[0]['start'], reverse=True)
    
    marked_text = text
    for entity, marker in entities:
        start = entity['start']
        end = entity['end']
        entity_text = text[start:end]
        marked_text = (
            marked_text[:start] + 
            f"[{marker}] {entity_text} [/{marker}]" + 
            marked_text[end:]
        )
    
    return marked_text


def create_re_samples(documents: List[Dict], include_negative: bool = True) -> List[Dict]:
    """
    Create relation extraction samples from documents.
    
    For each document:
    1. Find all entity pairs involved in relations (positive samples)
    2. Create negative samples from entity pairs without relations
    """
    samples = []
    relation_types = set()
    
    for doc in documents:
        full_text = doc['title'] + ' ' + doc['abstract']
        
        # Create entity lookup by identifier
        entity_by_id = defaultdict(list)
        for entity in doc['entities']:
            # Handle multiple identifiers (comma-separated)
            identifiers = entity['identifier'].split(',')
            for ident in identifiers:
                ident = ident.strip()
                if ident:
                    entity_by_id[ident].append(entity)
        
        # Process relations (positive samples)
        positive_pairs = set()
        for relation in doc['relations']:
            rel_type = relation['type']
            arg1_id = relation['arg1']
            arg2_id = relation['arg2']
            
            relation_types.add(rel_type)
            
            # Get entities for this relation
            arg1_entities = entity_by_id.get(arg1_id, [])
            arg2_entities = entity_by_id.get(arg2_id, [])
            
            # Create samples for each combination
            for e1 in arg1_entities:
                for e2 in arg2_entities:
                    if e1 == e2:
                        continue
                    
                    # Create marked text
                    marked_text = mark_entities_in_text(full_text, e1, e2)
                    
                    samples.append({
                        'doc_id': doc['id'],
                        'text': marked_text,
                        'entity1': e1['text'],
                        'entity1_type': e1['type'],
                        'entity2': e2['text'],
                        'entity2_type': e2['type'],
                        'relation': rel_type,
                        'label': rel_type
                    })
                    
                    # Track positive pairs
                    pair_key = (e1['start'], e1['end'], e2['start'], e2['end'])
                    positive_pairs.add(pair_key)
        
        # Create negative samples (entity pairs without relations)
        if include_negative:
            entities = doc['entities']
            negative_count = 0
            max_negative = len(doc['relations']) * 2  # Limit negatives
            
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if i >= j:
                        continue
                    
                    pair_key = (e1['start'], e1['end'], e2['start'], e2['end'])
                    reverse_key = (e2['start'], e2['end'], e1['start'], e1['end'])
                    
                    # Skip if this pair has a relation
                    if pair_key in positive_pairs or reverse_key in positive_pairs:
                        continue
                    
                    # Only create negative for certain entity type combinations
                    valid_pairs = [
                        ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
                        ('GeneOrGeneProduct', 'DiseaseOrPhenotypicFeature'),
                        ('ChemicalEntity', 'GeneOrGeneProduct'),
                        ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
                        ('SequenceVariant', 'DiseaseOrPhenotypicFeature'),
                    ]
                    
                    type_pair = (e1['type'], e2['type'])
                    reverse_type_pair = (e2['type'], e1['type'])
                    
                    if type_pair not in valid_pairs and reverse_type_pair not in valid_pairs:
                        continue
                    
                    if negative_count >= max_negative:
                        break
                    
                    marked_text = mark_entities_in_text(full_text, e1, e2)
                    
                    samples.append({
                        'doc_id': doc['id'],
                        'text': marked_text,
                        'entity1': e1['text'],
                        'entity1_type': e1['type'],
                        'entity2': e2['text'],
                        'entity2_type': e2['type'],
                        'relation': 'NO_RELATION',
                        'label': 'NO_RELATION'
                    })
                    
                    negative_count += 1
    
    return samples, relation_types


def save_jsonl(samples: List[Dict], filepath: str):
    """Save samples in JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def main():
    data_dir = Path("BIORED")
    output_dir = Path("data/re")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_relation_types = set()
    
    print("\n" + "="*60)
    print("  Preparing Relation Extraction Data")
    print("="*60)
    
    for split in ['Train', 'Dev', 'Test']:
        print(f"\nProcessing {split}...")
        input_file = data_dir / f"{split}.PubTator"
        
        # Parse documents
        documents = parse_pubtator_file(input_file)
        print(f"  Loaded {len(documents)} documents")
        
        # Create RE samples
        samples, relation_types = create_re_samples(documents)
        all_relation_types.update(relation_types)
        
        # Count by label
        label_counts = defaultdict(int)
        for sample in samples:
            label_counts[sample['label']] += 1
        
        print(f"  Created {len(samples)} samples:")
        for label, count in sorted(label_counts.items()):
            print(f"    • {label}: {count}")
        
        # Save
        output_file = output_dir / f"{split.lower()}.jsonl"
        save_jsonl(samples, output_file)
        print(f"  Saved to {output_file}")
    
    # Save label list
    all_labels = sorted(list(all_relation_types)) + ['NO_RELATION']
    with open(output_dir / "labels.txt", 'w', encoding='utf-8') as f:
        for label in all_labels:
            f.write(f"{label}\n")
    
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"\nRelation types found: {len(all_labels)}")
    for label in all_labels:
        print(f"  • {label}")
    
    print(f"\n✅ RE data preparation complete!")
    print(f"   Output directory: {output_dir}")


if __name__ == "__main__":
    main()
