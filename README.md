# 🧬 Biomedical Knowledge Graph Construction using BioBERT with LoRA

## Project Overview

This project implements a complete pipeline for building a **Biomedical Knowledge Graph** from scientific literature using fine-tuned BioBERT models with LoRA (Low-Rank Adaptation) for efficient training. The pipeline includes:

1. **Named Entity Recognition (NER)** - Identify biomedical entities in text
2. **Relation Extraction (RE)** - Detect relationships between entities
3. **Knowledge Graph Construction** - Build queryable knowledge graphs

---

## 📊 Dataset: BIORED

**BIORED (Biomedical Relation Extraction Dataset)** is a comprehensive dataset sourced from PubMed abstracts, designed for both NER and RE tasks in the biomedical domain.

### Dataset Statistics

| Split | Documents | Entities | Relations | Avg Entities/Doc | Avg Relations/Doc |
|-------|-----------|----------|-----------|------------------|-------------------|
| **Train** | 400 | 13,351 | 4,178 | 33.4 | 10.4 |
| **Dev** | 100 | 3,533 | 1,162 | 35.3 | 11.6 |
| **Test** | 100 | 3,535 | 1,163 | 35.4 | 11.6 |
| **Total** | **600** | **20,419** | **6,503** | **34.0** | **10.8** |

### Entity Types (6 Categories)

| Entity Type | Count | Percentage | Description |
|-------------|-------|------------|-------------|
| **GeneOrGeneProduct** | 6,697 | 33.2% | Genes, proteins, gene products |
| **DiseaseOrPhenotypicFeature** | 5,545 | 27.3% | Diseases, symptoms, phenotypes |
| **ChemicalEntity** | 4,429 | 21.4% | Drugs, compounds, chemicals |
| **OrganismTaxon** | 2,192 | 10.7% | Species (humans, mice, etc.) |
| **SequenceVariant** | 1,381 | 6.7% | Genetic mutations, SNPs |
| **CellLine** | 175 | 0.8% | Cell lines used in research |

### Relation Types (8 Categories)

| Relation Type | Count | Percentage | Description |
|---------------|-------|------------|-------------|
| **Association** | 3,387 | 52.5% | General association between entities |
| **Positive_Correlation** | 1,766 | 26.1% | Entity A increases with entity B |
| **Negative_Correlation** | 1,150 | 18.3% | Entity A decreases with entity B |
| **Bind** | 89 | 1.5% | Physical binding interaction |
| **Cotreatment** | 55 | 0.7% | Combined treatment effects |
| **Comparison** | 39 | 0.7% | Comparative relationships |
| **Drug_Interaction** | 13 | 0.3% | Drug-drug interactions |
| **Conversion** | 4 | 0.1% | Chemical/biological conversion |

---

## 🎯 Project Phases

### **Phase 1: Data Preparation & Preprocessing**
- ✅ Download and parse BIORED dataset (PubTator format)
- ✅ Convert to IOB format for NER (sequence tagging)
- ✅ Create entity pairs for RE (classification)
- ✅ Dataset analysis and statistics generation

### **Phase 2: Model Fine-Tuning (BioBERT + LoRA)**
- 🔄 Fine-tune BioBERT for NER with LoRA adapters
- 🔄 Fine-tune BioBERT for RE with LoRA adapters
- 🔄 Hyperparameter optimization
- 🔄 Model evaluation (Precision, Recall, F1)

### **Phase 3: Information Extraction & KG Construction**
- 🔄 Apply NER model to extract entities from text
- 🔄 Apply RE model to identify relationships
- 🔄 Generate structured triples (Entity1, Relation, Entity2)
- 🔄 Set up Neo4j graph database
- 🔄 Load triples into knowledge graph

### **Phase 4: Evaluation & Deployment**
- 🔄 Benchmark against published results
- 🔄 Build interactive query interface
- 🔄 Visualization with NetworkX/pyvis
- 🔄 Deploy demo application (Streamlit/Flask)

---

## 🛠️ Technology Stack

### Core Framework
- **Base Model**: BioBERT (Biomedical BERT)
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch + Hugging Face Transformers

### Libraries
```
transformers >= 4.30.0
peft >= 0.5.0
datasets >= 2.14.0
torch >= 2.0.0
scikit-learn >= 1.3.0
seqeval >= 1.2.2
```

### Knowledge Graph
- **Database**: Neo4j
- **Visualization**: NetworkX, pyvis
- **Interface**: Streamlit or Flask

---

## 📁 Project Structure

```
Project/
├── BIORED/                      # Raw dataset
│   ├── Train.PubTator
│   ├── Dev.PubTator
│   └── Test.PubTator
│
├── data/                        # Processed data
│   ├── ner/                     # NER formatted data
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   ├── test.jsonl
│   │   └── labels.txt
│   └── re/                      # RE formatted data
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
│
├── models/                      # Trained models
│   ├── biobert-ner-lora/
│   └── biobert-re-lora/
│
├── scripts/                     # Training & inference
│   ├── prepare_ner_data.py
│   ├── prepare_re_data.py
│   ├── train_ner.py
│   ├── train_re.py
│   └── inference.py
│
├── knowledge_graph/             # KG construction
│   ├── build_kg.py
│   └── query_kg.py
│
├── parse_biored_ner.py         # NER data parser
├── analyze_dataset.py          # Dataset analysis
├── dataset_statistics.json     # Statistics output
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install transformers peft datasets torch scikit-learn seqeval
```

### 2. Analyze Dataset
```bash
python analyze_dataset.py
```

### 3. Prepare Data for NER
```bash
python parse_biored_ner.py
```

### 4. Fine-tune BioBERT with LoRA
```bash
# Train NER model
python scripts/train_ner.py

# Train RE model
python scripts/train_re.py
```

### 5. Build Knowledge Graph
```bash
python knowledge_graph/build_kg.py
```

---

## 📈 Expected Results

### Baseline Performance (from literature)
- **NER F1 Score**: ~85-90%
- **RE F1 Score**: ~75-85%

### LoRA Benefits
- ✅ **90% reduction** in trainable parameters
- ✅ **50% faster** training time
- ✅ **Lower GPU memory** requirements
- ✅ Comparable performance to full fine-tuning

---

## 🧪 Use Cases

### 1. Drug Discovery
**Query**: "What chemicals are associated with Alzheimer's disease?"
```cypher
MATCH (c:Chemical)-[r:Association|Positive_Correlation]->(d:Disease {name: "Alzheimer"})
RETURN c.name, type(r), d.name
```

### 2. Gene-Disease Associations
**Query**: "Which genes are correlated with diabetes?"
```cypher
MATCH (g:Gene)-[r:Positive_Correlation|Association]->(d:Disease {name: "Diabetes"})
RETURN g.name, r.type, d.name
```

### 3. Drug Interactions
**Query**: "Find all drug-drug interactions"
```cypher
MATCH (c1:Chemical)-[r:Drug_Interaction]->(c2:Chemical)
RETURN c1.name, c2.name
```

---

## 📚 References

### Dataset
- **BIORED**: Luo et al. (2022) - [Paper](https://academic.oup.com/bib/article/23/5/bbac282/6645993)

### Models
- **BioBERT**: Lee et al. (2020) - Biomedical language representation model
- **LoRA**: Hu et al. (2021) - Low-Rank Adaptation of Large Language Models

### Knowledge Graph
- **Neo4j**: Graph database for KG storage and querying
- **Cypher**: Graph query language

---

## 👥 Team

**Course**: Natural Language Processing (NLP)  
**Semester**: 6  
**Institution**: CAI  
**Year**: 2025

---

## 📝 License

This project is for educational purposes as part of the NLP course curriculum.

---

## 🔗 Next Steps

1. ✅ Dataset analysis complete
2. 🔄 Implement NER data preparation
3. 🔄 Implement RE data preparation
4. 🔄 Fine-tune BioBERT with LoRA
5. 🔄 Build and query knowledge graph
6. 🔄 Deploy interactive demo

---

**Last Updated**: December 11, 2025
