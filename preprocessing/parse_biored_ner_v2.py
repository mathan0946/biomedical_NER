"""
Parse BIORED PubTator format to IOB tagging format for NER task.
Correctly handles character offset to token mapping.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json


def parse_pubtator_file(filepath: str) -> List[Dict]:
    """
    Parse PubTator format file and extract documents with annotations.
    """
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
                
                # Entity annotation: PMID\tstart\tend\ttext\ttype\tid
                if len(parts) >= 5 and parts[1].isdigit():
                    if current_doc:
                        current_doc['entities'].append({
                            'start': int(parts[1]),
                            'end': int(parts[2]),
                            'text': parts[3],
                            'type': parts[4],
                            'identifier': parts[5] if len(parts) > 5 else ''
                        })
                # Relation line
                elif len(parts) >= 4:
                    if current_doc:
                        current_doc['relations'].append({
                            'type': parts[1],
                            'arg1': parts[2],
                            'arg2': parts[3],
                            'novel': parts[4] if len(parts) > 4 else 'No'
                        })
    
    if current_doc:
        documents.append(current_doc)
    
    return documents


def tokenize_text_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text and return tokens with character offsets.
    Uses regex to properly handle punctuation and special characters.
    """
    tokens = []
    # Pattern to match words, numbers, and punctuation separately
    pattern = r'\S+'
    
    for match in re.finditer(pattern, text):
        token = match.group()
        start = match.start()
        end = match.end()
        tokens.append((token, start, end))
    
    return tokens


def create_iob_tags(text: str, entities: List[Dict]) -> List[Tuple[str, str]]:
    """
    Create IOB tags for a text given entity annotations.
    Handles overlapping entities by preferring the longest match.
    """
    tokens = tokenize_text_with_offsets(text)
    
    # Sort entities by start position, then by length (longer first)
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    # Create character-level labels
    char_labels = ['O'] * len(text)
    
    for entity in sorted_entities:
        ent_start = entity['start']
        ent_end = entity['end']
        ent_type = entity['type']
        
        # Skip if out of bounds
        if ent_start >= len(text) or ent_end > len(text):
            continue
            
        # Only label if not already labeled (first entity wins for overlaps)
        if all(char_labels[i] == 'O' for i in range(ent_start, min(ent_end, len(text)))):
            # Mark beginning
            char_labels[ent_start] = f'B-{ent_type}'
            # Mark inside
            for i in range(ent_start + 1, min(ent_end, len(text))):
                char_labels[i] = f'I-{ent_type}'
    
    # Convert character labels to token labels
    iob_tags = []
    for token, start, end in tokens:
        # Get the label for this token based on its first character
        if start < len(char_labels):
            label = char_labels[start]
            # If the first character is 'I-', check if previous token was from same entity
            # If not, change to 'B-'
            if label.startswith('I-'):
                if iob_tags:
                    prev_label = iob_tags[-1][1]  # Get label from tuple
                    if not (prev_label.endswith(label[2:]) and 
                           (prev_label.startswith('B-') or prev_label.startswith('I-'))):
                        label = 'B-' + label[2:]
                else:
                    label = 'B-' + label[2:]
            iob_tags.append((token, label))
        else:
            iob_tags.append((token, 'O'))
    
    return iob_tags


def convert_documents_to_iob(documents: List[Dict]) -> List[List[Tuple[str, str]]]:
    """
    Convert all documents to IOB format.
    """
    iob_data = []
    
    for doc in documents:
        # Combine title and abstract with newline separator
        title = doc['title']
        abstract = doc['abstract']
        
        # IMPORTANT: In PubTator, title is followed by newline, then abstract
        # So abstract offsets start after title_length + 1 (for newline)
        # But actually it seems like they concatenate with just different offset tracking
        
        # Looking at the data, entity offsets in abstract are relative to the full document
        # The title ends at position len(title), then there's effectively a newline
        # Abstract starts at position len(title) + 1
        full_text = title + '\n' + abstract if abstract else title
        
        # Convert to IOB
        iob_tags = create_iob_tags(full_text, doc['entities'])
        
        if iob_tags:
            iob_data.append(iob_tags)
    
    return iob_data


def save_conll_format(iob_data: List[List[Tuple[str, str]]], output_path: str):
    """Save in CoNLL format (token label per line, blank line between sentences)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in iob_data:
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            f.write("\n")


def save_jsonl_format(iob_data: List[List[Tuple[str, str]]], output_path: str):
    """Save in JSONL format for HuggingFace datasets."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in iob_data:
            tokens = [t[0] for t in sentence]
            labels = [t[1] for t in sentence]
            f.write(json.dumps({"tokens": tokens, "ner_tags": labels}) + '\n')


def collect_labels(iob_data: List[List[Tuple[str, str]]]) -> set:
    """Collect all unique labels from the data."""
    labels = set()
    for sentence in iob_data:
        for _, label in sentence:
            labels.add(label)
    return labels


def main():
    # Define paths
    data_dir = Path("BIORED")
    output_dir = Path("data/ner")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {
        'train': 'Train.PubTator',
        'dev': 'Dev.PubTator',
        'test': 'Test.PubTator'
    }
    
    all_labels = set()
    
    for split_name, filename in splits.items():
        filepath = data_dir / filename
        print(f"\nProcessing {filepath}...")
        
        # Parse documents
        documents = parse_pubtator_file(str(filepath))
        print(f"  Found {len(documents)} documents")
        
        # Count entities before conversion
        total_entities = sum(len(doc['entities']) for doc in documents)
        print(f"  Total entities: {total_entities}")
        
        # Convert to IOB
        iob_data = convert_documents_to_iob(documents)
        
        # Count entity tokens
        entity_tokens = sum(
            1 for sent in iob_data for _, label in sent if label != 'O'
        )
        print(f"  Entity tokens after conversion: {entity_tokens}")
        
        # Collect labels
        labels = collect_labels(iob_data)
        all_labels.update(labels)
        print(f"  Labels found: {sorted(labels)}")
        
        # Save in both formats
        save_conll_format(iob_data, str(output_dir / f"{split_name}.txt"))
        save_jsonl_format(iob_data, str(output_dir / f"{split_name}.jsonl"))
        
        print(f"  Saved {len(iob_data)} examples")
        
        # Print first example for verification
        if iob_data:
            print(f"\n  First example preview:")
            first_sent = iob_data[0]
            for i, (token, label) in enumerate(first_sent[:30]):  # First 30 tokens
                if label != 'O':
                    print(f"    {token}: {label}")
    
    # Save labels
    labels_sorted = sorted(all_labels)
    with open(output_dir / "labels.txt", 'w') as f:
        for label in labels_sorted:
            f.write(f"{label}\n")
    
    print(f"\n\nAll labels ({len(labels_sorted)}):")
    for label in labels_sorted:
        print(f"  {label}")


if __name__ == "__main__":
    main()
