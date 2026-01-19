# Preprocessing Scripts

This folder contains the final preprocessing scripts used to prepare the BIORED dataset for training.

## Scripts

### 1. `parse_biored_ner_v2.py`
**Purpose**: Parse BIORED PubTator format to IOB (Inside-Outside-Beginning) format for NER training

**Input**: 
- `BIORED/Train.PubTator`
- `BIORED/Dev.PubTator`
- `BIORED/Test.PubTator`

**Output**: 
- `data/ner/train.jsonl` - 600 documents
- `data/ner/dev.jsonl` - 100 documents
- `data/ner/test.jsonl` - 100 documents
- `data/ner/labels.txt` - 13 labels (B-/I- for 6 entity types + O)

**Key Features**:
- Character offset-based entity alignment (fixed in v2)
- Handles multi-word entities correctly
- Produces IOB-tagged tokens
- Result: 16,590+ entity tokens in training data

**Usage**:
```bash
python preprocessing/parse_biored_ner_v2.py
```

---

### 2. `prepare_re_data.py`
**Purpose**: Parse BIORED dataset to entity-pair format for Relation Extraction training

**Input**: 
- `BIORED/Train.PubTator`
- `BIORED/Dev.PubTator`
- `BIORED/Test.PubTator`

**Output**: 
- `data/re/train.jsonl` - ~30,000+ entity pairs
- `data/re/dev.jsonl` - ~5,000 entity pairs
- `data/re/test.jsonl` - ~5,000 entity pairs
- `data/re/labels.txt` - 9 labels (8 relation types + No_Relation)

**Key Features**:
- Extracts entity pairs from documents
- Marks entities with special tokens: `[E1]`, `[/E1]`, `[E2]`, `[/E2]`
- Includes entity type information
- Handles 8 relation types: Association, Bind, Comparison, Conversion, Cotreatment, Drug_Interaction, Negative_Correlation, Positive_Correlation

**Usage**:
```bash
python preprocessing/prepare_re_data.py
```

**Example Output**:
```json
{
  "doc_id": "123456",
  "text": "[E1] Metformin [/E1] activates [E2] AMPK [/E2]",
  "entity1": "Metformin",
  "entity1_type": "ChemicalEntity",
  "entity2": "AMPK",
  "entity2_type": "GeneOrGeneProduct",
  "relation": "Positive_Correlation",
  "label": "Positive_Correlation"
}
```

---

### 3. `analyze_dataset.py`
**Purpose**: Analyze BIORED dataset statistics

**Input**: 
- `BIORED/Train.PubTator`
- `BIORED/Dev.PubTator`
- `BIORED/Test.PubTator`

**Output**: 
- `dataset_statistics.json` - Detailed statistics
- Console output with summary

**Statistics Provided**:
- Number of documents per split
- Entity type distribution
- Relation type distribution
- Average entities per document
- Average relations per document

**Usage**:
```bash
python preprocessing/analyze_dataset.py
```

---

## Dataset Format

### BIORED PubTator Format
```
12345|t|Title of the paper
12345|a|Abstract text goes here.
12345	0	9	Metformin	ChemicalEntity	MESH:D008687
12345	20	24	AMPK	GeneOrGeneProduct	NCBIGene:5563
12345	CID	MESH:D008687	NCBIGene:5563	Positive_Correlation
```

### NER Output Format (IOB)
```json
{
  "tokens": ["Metformin", "activates", "AMPK", "."],
  "ner_tags": ["B-ChemicalEntity", "O", "B-GeneOrGeneProduct", "O"]
}
```

### RE Output Format (Entity Pairs)
```json
{
  "doc_id": "12345",
  "text": "[E1] Metformin [/E1] activates [E2] AMPK [/E2]",
  "entity1": "Metformin",
  "entity1_type": "ChemicalEntity",
  "entity2": "AMPK",
  "entity2_type": "GeneOrGeneProduct",
  "relation": "Positive_Correlation",
  "label": "Positive_Correlation"
}
```

---

## Entity Types (6)
1. **GeneOrGeneProduct** - Genes, proteins, enzymes
2. **DiseaseOrPhenotypicFeature** - Diseases, symptoms, phenotypes
3. **ChemicalEntity** - Drugs, compounds, chemicals
4. **SequenceVariant** - Mutations, SNPs, genetic variants
5. **OrganismTaxon** - Species, organisms
6. **CellLine** - Cell lines (HeLa, HEK293, etc.)

---

## Relation Types (8)
1. **Association** - General association between entities
2. **Positive_Correlation** - Positive relationship (activation, increase)
3. **Negative_Correlation** - Negative relationship (inhibition, decrease)
4. **Bind** - Physical binding/interaction
5. **Cotreatment** - Combined treatment effects
6. **Drug_Interaction** - Drug-drug interactions
7. **Comparison** - Comparative relationships
8. **Conversion** - Transformation/conversion relationships

---

## Pipeline

1. **Download BIORED dataset** → Place in `BIORED/` folder
2. **Run NER preprocessing** → `python preprocessing/parse_biored_ner_v2.py`
3. **Run RE preprocessing** → `python preprocessing/prepare_re_data.py`
4. **Analyze dataset** → `python preprocessing/analyze_dataset.py`
5. **Train models** → `python train_ner.py` and `python train_re.py`

---

## Notes

- **parse_biored_ner.py** was the initial version with entity alignment issues (F1=0%)
- **parse_biored_ner_v2.py** is the corrected version with proper character offset alignment (F1=86%)
- All scripts use UTF-8 encoding
- Output files are in JSONL format (one JSON object per line)
- The preprocessing creates balanced datasets suitable for fine-tuning BioBERT with LoRA

---

## Results

After preprocessing and training:
- **NER Model**: 86.1% F1 score (Precision: 84.5%, Recall: 87.8%)
- **RE Model**: 49.7% F1 score (initial), improved to 55-60% F1 with enhancements
