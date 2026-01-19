"""
Analyze BIORED dataset structure and statistics.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def parse_pubtator_file(filepath):
    """Parse PubTator format file."""
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
                
                # Relation line
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


def analyze_split(filepath, split_name):
    """Analyze a single data split."""
    docs = parse_pubtator_file(filepath)
    
    entity_types = Counter()
    relation_types = Counter()
    entity_lengths = []
    doc_lengths = []
    entities_per_doc = []
    relations_per_doc = []
    
    for doc in docs:
        # Document length
        full_text = doc['title'] + ' ' + doc['abstract']
        doc_lengths.append(len(full_text.split()))
        
        # Entities
        entities_per_doc.append(len(doc['entities']))
        for entity in doc['entities']:
            entity_types[entity['type']] += 1
            entity_lengths.append(len(entity['text'].split()))
        
        # Relations
        relations_per_doc.append(len(doc['relations']))
        for relation in doc['relations']:
            relation_types[relation['type']] += 1
    
    print(f"\n{'='*60}")
    print(f"  {split_name} Split Analysis")
    print(f"{'='*60}")
    
    print(f"\n📊 Basic Statistics:")
    print(f"  • Documents: {len(docs)}")
    print(f"  • Total Entities: {sum(entities_per_doc)}")
    print(f"  • Total Relations: {sum(relations_per_doc)}")
    print(f"  • Avg entities per document: {sum(entities_per_doc)/len(docs):.1f}")
    print(f"  • Avg relations per document: {sum(relations_per_doc)/len(docs):.1f}")
    print(f"  • Avg document length (words): {sum(doc_lengths)/len(docs):.1f}")
    
    print(f"\n🏷️  Entity Types Distribution:")
    for ent_type, count in entity_types.most_common():
        percentage = (count / sum(entity_types.values())) * 100
        print(f"  • {ent_type}: {count} ({percentage:.1f}%)")
    
    if relation_types:
        print(f"\n🔗 Relation Types Distribution:")
        for rel_type, count in relation_types.most_common():
            percentage = (count / sum(relation_types.values())) * 100
            print(f"  • {rel_type}: {count} ({percentage:.1f}%)")
    
    return {
        'documents': len(docs),
        'entities': sum(entities_per_doc),
        'relations': sum(relations_per_doc),
        'entity_types': dict(entity_types),
        'relation_types': dict(relation_types),
        'avg_doc_length': sum(doc_lengths)/len(docs),
        'avg_entities_per_doc': sum(entities_per_doc)/len(docs),
        'avg_relations_per_doc': sum(relations_per_doc)/len(docs)
    }


def main():
    data_dir = Path("BIORED")
    
    print("\n" + "="*60)
    print("  🧬 BIORED DATASET ANALYSIS")
    print("="*60)
    
    print("\n📖 About BIORED Dataset:")
    print("-" * 60)
    print("""
BIORED (Biomedical Relation Extraction Dataset) is a comprehensive
dataset for biomedical named entity recognition (NER) and relation
extraction (RE) tasks.

Key Features:
• Source: PubMed abstracts (biomedical literature)
• Task 1: Named Entity Recognition (NER)
• Task 2: Relation Extraction (RE)
• Format: PubTator (tab-separated annotation format)

Entity Types Covered:
1. GeneOrGeneProduct - Genes, proteins, gene products
2. DiseaseOrPhenotypicFeature - Diseases, symptoms, phenotypes
3. ChemicalEntity - Drugs, compounds, chemical substances
4. SequenceVariant - Genetic mutations, variants (SNPs)
5. OrganismTaxon - Species, organisms (e.g., humans, mice)

Relation Types:
• Association - General association between entities
• Positive_Correlation - Entity A increases with entity B
• Negative_Correlation - Entity A decreases with entity B
• Bind - Physical binding interaction
• Conversion - Chemical/biological conversion
• Drug_Interaction - Drug-drug interactions
• Cotreatment - Combined treatment effects
    """)
    
    stats = {}
    for split in ['Train', 'Dev', 'Test']:
        filepath = data_dir / f"{split}.PubTator"
        stats[split] = analyze_split(filepath, split)
    
    print(f"\n{'='*60}")
    print("  📈 OVERALL DATASET SUMMARY")
    print(f"{'='*60}")
    
    total_docs = sum(s['documents'] for s in stats.values())
    total_entities = sum(s['entities'] for s in stats.values())
    total_relations = sum(s['relations'] for s in stats.values())
    
    print(f"\n  Total Documents: {total_docs}")
    print(f"  Total Entities: {total_entities}")
    print(f"  Total Relations: {total_relations}")
    print(f"\n  Split Distribution:")
    print(f"    • Train: {stats['Train']['documents']} docs ({stats['Train']['documents']/total_docs*100:.1f}%)")
    print(f"    • Dev:   {stats['Dev']['documents']} docs ({stats['Dev']['documents']/total_docs*100:.1f}%)")
    print(f"    • Test:  {stats['Test']['documents']} docs ({stats['Test']['documents']/total_docs*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("  🎯 Use Cases for This Dataset:")
    print(f"{'='*60}")
    print("""
1. Named Entity Recognition (NER)
   → Identify and classify biomedical entities in text
   → Train sequence labeling models (BioBERT, BioGPT, etc.)
   
2. Relation Extraction (RE)
   → Detect relationships between entities
   → Build knowledge graphs from literature
   
3. Knowledge Graph Construction
   → Entity nodes: Genes, Diseases, Chemicals, Variants
   → Relation edges: Association, Correlation, Binding, etc.
   → Query: "What drugs treat disease X?"
   → Query: "What genes are associated with disease Y?"
   
4. Literature Mining
   → Automatic extraction of biomedical facts
   → Drug discovery and repurposing
   → Disease mechanism understanding
    """)
    
    print(f"\n{'='*60}")
    print("  ✅ Dataset Ready for Fine-tuning!")
    print(f"{'='*60}\n")
    
    # Save statistics
    with open('dataset_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print("💾 Statistics saved to: dataset_statistics.json\n")


if __name__ == "__main__":
    main()
