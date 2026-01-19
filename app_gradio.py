"""
Biomedical NER + RE Pipeline with Knowledge Graph - Gradio Web Interface
Extracts biomedical entities AND their relationships from text/PDF.
Visualizes as Knowledge Graph with labeled edges.
"""

import gradio as gr
import torch
import json
import re
import io
import base64
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from peft import PeftModel

try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# ============================================================================
# Configuration
# ============================================================================

ENTITY_COLORS = {
    "GeneOrGeneProduct": "#e74c3c",
    "DiseaseOrPhenotypicFeature": "#1abc9c",
    "ChemicalEntity": "#3498db",
    "SequenceVariant": "#27ae60",
    "CellLine": "#f39c12",
    "OrganismTaxon": "#9b59b6",
}

SHORT_NAMES = {
    "GeneOrGeneProduct": "Gene",
    "DiseaseOrPhenotypicFeature": "Disease",
    "ChemicalEntity": "Chemical",
    "SequenceVariant": "Variant",
    "CellLine": "CellLine",
    "OrganismTaxon": "Organism",
}

RELATION_COLORS = {
    "Positive_Correlation": "#27ae60",
    "Negative_Correlation": "#e74c3c",
    "Association": "#7f8c8d",
    "Bind": "#9b59b6",
    "Cotreatment": "#3498db",
    "Drug_Interaction": "#e67e22",
    "Comparison": "#95a5a6",
    "Conversion": "#1abc9c",
}

RELATION_STYLES = {
    "Positive_Correlation": "-",
    "Negative_Correlation": "--",
    "Association": ":",
    "Bind": "-",
    "Cotreatment": "-.",
    "Drug_Interaction": "-",
    "Comparison": ":",
    "Conversion": "-",
}


# ============================================================================
# NER Pipeline
# ============================================================================

class BiomedicalNERPipeline:
    """Pipeline for biomedical named entity recognition."""
    
    def __init__(self, model_path: str = "models/biobert-ner-lora"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned NER model."""
        print(f"Loading NER model from {self.model_path}...")
        
        with open(self.model_path / "id2label.json", "r") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        with open(self.model_path / "label2id.json", "r") as f:
            self.label2id = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        base_model = AutoModelForTokenClassification.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.2",
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"NER model loaded on {self.device}")
    
    def predict(self, text: str):
        """Extract entities from text."""
        if not text.strip():
            return []
        
        # Word tokenization with offsets
        words = []
        word_offsets = []
        for match in re.finditer(r'\S+', text):
            words.append(match.group())
            word_offsets.append((match.start(), match.end()))
        
        if not words:
            return []
        
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        
        word_ids = inputs.word_ids()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        # Align predictions with words
        word_labels = []
        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx and word_idx < len(words):
                label = self.id2label.get(predictions[idx], "O")
                word_labels.append((word_idx, label))
            prev_word_idx = word_idx
        
        # Extract entities
        entities = []
        current_entity = None
        
        for word_idx, label in word_labels:
            word = words[word_idx]
            start, end = word_offsets[word_idx]
            
            if label.startswith("B-"):
                if current_entity:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "color": ENTITY_COLORS.get(entity_type, "#808080"),
                    "short_type": SHORT_NAMES.get(entity_type, entity_type),
                }
            elif label.startswith("I-") and current_entity:
                if label[2:] == current_entity["type"]:
                    current_entity["end"] = end
                else:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                    entity_type = label[2:]
                    current_entity = {
                        "type": entity_type,
                        "start": start,
                        "end": end,
                        "color": ENTITY_COLORS.get(entity_type, "#808080"),
                        "short_type": SHORT_NAMES.get(entity_type, entity_type),
                    }
            else:
                if current_entity:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
            entities.append(current_entity)
        
        return [e for e in entities if e.get("text", "").strip()]


# ============================================================================
# Relation Extraction Pipeline
# ============================================================================

class RelationExtractionPipeline:
    """Pipeline for relation extraction between entities."""
    
    def __init__(self, model_path: str = "models/biobert-re-lora"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned RE model."""
        print(f"Loading RE model from {self.model_path}...")
        
        with open(self.model_path / "id2label.json", "r") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        with open(self.model_path / "label2id.json", "r") as f:
            self.label2id = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.2",
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
        
        inputs = self.tokenizer(
            marked_text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
        
        relation = self.id2label[pred_id]
        return relation, confidence


# ============================================================================
# Knowledge Graph with Real Relations
# ============================================================================

def extract_relations(text: str, entities: list, re_pipeline, max_pairs: int = 20) -> list:
    """Extract relations between entity pairs using RE model."""
    if len(entities) < 2:
        return []
    
    # Deduplicate entities
    unique_entities = {}
    for e in entities:
        key = (e["text"].lower(), e["type"])
        if key not in unique_entities:
            unique_entities[key] = e
    
    entities = list(unique_entities.values())
    
    # Get entity pairs (limit for performance)
    pairs = list(combinations(entities, 2))[:max_pairs]
    
    relations = []
    for e1, e2 in pairs:
        try:
            relation, confidence = re_pipeline.predict(text, e1, e2)
            
            # Keep meaningful relations with confidence > 0.35
            if relation != "NO_RELATION" and confidence > 0.35:
                relations.append({
                    "source": e1["text"],
                    "source_type": e1["type"],
                    "target": e2["text"],
                    "target_type": e2["type"],
                    "relation": relation,
                    "confidence": confidence,
                })
        except Exception as e:
            continue
    
    return relations


def create_knowledge_graph_with_relations(entities: list, relations: list) -> str:
    """Create Knowledge Graph with labeled relationship edges."""
    if not HAS_NETWORKX or not entities:
        return "<p>No entities found or NetworkX not available.</p>"
    
    G = nx.Graph()
    
    # Add entity nodes (deduplicated)
    unique_entities = {}
    for e in entities:
        key = e["text"]
        if key not in unique_entities:
            unique_entities[key] = e
            short_name = e["text"][:20] + "..." if len(e["text"]) > 20 else e["text"]
            G.add_node(
                short_name,
                full_text=e["text"],
                entity_type=e["type"],
                color=ENTITY_COLORS.get(e["type"], "#808080"),
            )
    
    # Add relation edges
    for rel in relations:
        source = rel["source"][:20] + "..." if len(rel["source"]) > 20 else rel["source"]
        target = rel["target"][:20] + "..." if len(rel["target"]) > 20 else rel["target"]
        
        if source in G.nodes() and target in G.nodes():
            G.add_edge(
                source, target,
                relation=rel["relation"],
                confidence=rel["confidence"],
                color=RELATION_COLORS.get(rel["relation"], "#7f8c8d"),
            )
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "No entities found", ha='center', va='center', fontsize=16)
        ax.axis('off')
    else:
        # Layout
        if len(G.nodes()) <= 8:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # Draw edges with colors based on relation type
        for (u, v, data) in G.edges(data=True):
            edge_color = data.get('color', '#7f8c8d')
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                edge_color=edge_color,
                width=2.5,
                alpha=0.7,
                ax=ax,
            )
        
        # Draw edge labels (relation types)
        edge_labels = {}
        for (u, v, data) in G.edges(data=True):
            rel = data.get('relation', '')
            conf = data.get('confidence', 0)
            # Shorten relation names
            short_rel = rel.replace('_Correlation', '').replace('_', ' ')
            edge_labels[(u, v)] = f"{short_rel}\n({conf:.0%})"
        
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels,
            font_size=7,
            font_color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
            ax=ax,
        )
        
        # Draw nodes
        node_colors = [G.nodes[n].get('color', '#808080') for n in G.nodes()]
        node_sizes = [1800 for _ in G.nodes()]
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            ax=ax,
        )
        
        # Create legends
        # Entity type legend
        entity_patches = []
        present_types = set(G.nodes[n]['entity_type'] for n in G.nodes())
        for etype, color in ENTITY_COLORS.items():
            if etype in present_types:
                short = SHORT_NAMES.get(etype, etype)
                entity_patches.append(mpatches.Patch(color=color, label=short))
        
        # Relation type legend
        relation_patches = []
        present_rels = set(data.get('relation') for u, v, data in G.edges(data=True))
        for rel, color in RELATION_COLORS.items():
            if rel in present_rels:
                short = rel.replace('_', ' ')
                relation_patches.append(mpatches.Patch(color=color, label=short))
        
        # Add legends
        if entity_patches:
            legend1 = ax.legend(
                handles=entity_patches,
                loc='upper left',
                title="Entity Types",
                fontsize=9,
                title_fontsize=10,
            )
            ax.add_artist(legend1)
        
        if relation_patches:
            ax.legend(
                handles=relation_patches,
                loc='upper right',
                title="Relationships",
                fontsize=9,
                title_fontsize=10,
            )
        
        ax.set_title("🧬 Biomedical Knowledge Graph", fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert to base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;">'


# ============================================================================
# PDF Processing
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file."""
    if pdf_file is None:
        return ""
    
    if not HAS_PDF:
        return "Error: pdfplumber not installed. Install with: pip install pdfplumber"
    
    try:
        text_parts = []
        with pdfplumber.open(pdf_file.name) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        text = "\n\n".join(text_parts)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:8000]  # Limit for performance
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# ============================================================================
# Initialize Models
# ============================================================================

print("\n" + "="*60)
print("  🧬 Biomedical Knowledge Graph Builder")
print("="*60)

ner_pipeline = BiomedicalNERPipeline()
re_pipeline = RelationExtractionPipeline()
print("\n✅ Both NER and RE models loaded!")


# ============================================================================
# Gradio Interface Functions
# ============================================================================

def create_highlighted_html(text: str, entities: list) -> str:
    """Create HTML with highlighted entities."""
    if not entities:
        return f"<p>{text}</p>"
    
    # Sort entities by position (reverse) to avoid offset issues
    sorted_entities = sorted(entities, key=lambda x: -x.get("start", 0))
    
    html = text
    for entity in sorted_entities:
        start = entity.get("start", 0)
        end = entity.get("end", 0)
        color = entity.get("color", "#808080")
        short_type = entity.get("short_type", "Entity")
        
        replacement = f'<mark style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 4px; margin: 1px;" title="{short_type}">{text[start:end]}</mark>'
        html = html[:start] + replacement + html[end:]
    
    return f'<div style="line-height: 2; font-size: 15px;">{html}</div>'


def format_relations_table(relations: list) -> str:
    """Format relations as markdown table."""
    if not relations:
        return "*No significant relationships found.*"
    
    md = "| Source | Relation | Target | Confidence |\n"
    md += "|--------|----------|--------|------------|\n"
    
    for r in relations:
        rel_name = r["relation"].replace("_", " ")
        md += f"| {r['source']} | **{rel_name}** | {r['target']} | {r['confidence']:.0%} |\n"
    
    return md


def process_text(text: str, extract_rels: bool = True):
    """Process text and extract entities + relations."""
    if not text.strip():
        return "Please enter some text.", "", [], "<p>No graph</p>", ""
    
    # Extract entities
    entities = ner_pipeline.predict(text)
    
    # Create highlighted HTML
    highlighted_html = create_highlighted_html(text, entities)
    
    # Create entity table
    seen = set()
    entity_data = []
    for e in entities:
        key = (e["text"], e["short_type"])
        if key not in seen:
            seen.add(key)
            entity_data.append([e["text"], e["short_type"]])
    
    # Summary
    entity_counts = defaultdict(int)
    for e in entities:
        entity_counts[e["short_type"]] += 1
    
    summary = f"### ✅ Found {len(entities)} entities ({len(entity_data)} unique)\n\n"
    for etype, count in sorted(entity_counts.items()):
        summary += f"- **{etype}**: {count}\n"
    
    # Extract relations if enabled
    relations = []
    relations_md = ""
    
    if extract_rels and len(entities) >= 2:
        summary += "\n\n*Extracting relationships...*\n"
        relations = extract_relations(text, entities, re_pipeline, max_pairs=25)
        
        if relations:
            summary += f"\n### 🔗 Found {len(relations)} relationships\n"
            relations_md = format_relations_table(relations)
        else:
            relations_md = "*No significant relationships found between entities.*"
    
    # Create Knowledge Graph with relations
    kg_html = create_knowledge_graph_with_relations(entities, relations)
    
    return summary, highlighted_html, entity_data, kg_html, relations_md


def process_pdf(pdf_file, extract_rels: bool = True):
    """Process PDF and extract entities + relations."""
    if pdf_file is None:
        return "Please upload a PDF file.", "", [], "", "<p>No graph</p>", ""
    
    text = extract_text_from_pdf(pdf_file)
    if text.startswith("Error"):
        return text, "", [], "", "<p>Error</p>", ""
    
    summary, highlighted, entity_data, kg_html, relations_md = process_text(text, extract_rels)
    preview = text[:2000] + "..." if len(text) > 2000 else text
    
    return summary, highlighted, entity_data, preview, kg_html, relations_md


# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(
    title="🧬 Biomedical Knowledge Graph",
    theme=gr.themes.Soft(),
) as demo:
    
    gr.Markdown("""
    # 🧬 Biomedical Knowledge Graph Builder
    
    **Extract entities AND their relationships from biomedical text using BioBERT + LoRA**
    
    This pipeline uses TWO fine-tuned models:
    1. **NER Model** (85% F1): Extracts 6 entity types
    2. **RE Model** (47% F1): Predicts 8 relationship types between entities
    
    | Entity Types | Relationship Types |
    |-------------|-------------------|
    | 🔴 Gene/Protein | ✅ Positive Correlation |
    | 🟢 Disease | ❌ Negative Correlation |
    | 🔵 Chemical/Drug | 🔗 Association |
    | 🟢 Variant | 🧲 Bind |
    | 🟡 Cell Line | 💊 Drug Interaction |
    | 🟣 Organism | 🔄 Cotreatment |
    """)
    
    with gr.Tabs():
        # Tab 1: Text Input
        with gr.TabItem("📝 Text Input"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Enter Biomedical Text",
                        placeholder="Paste biomedical text here...",
                        lines=8,
                    )
                    
                    extract_rels_cb = gr.Checkbox(
                        label="🔗 Extract Relationships (takes longer but shows real relations)",
                        value=True,
                    )
                    
                    gr.Examples(
                        examples=[
                            ["Metformin is a first-line drug for type 2 diabetes mellitus. It works by activating AMPK and inhibiting hepatic glucose production. The OCT1 transporter mediates its uptake into hepatocytes. Common variants like rs622342 in SLC22A1 affect drug response. Studies in HepG2 cells confirmed these findings."],
                            ["BRCA1 and BRCA2 mutations are associated with hereditary breast and ovarian cancer syndrome. Patients with these mutations have significantly increased lifetime risk. PARP inhibitors like Olaparib show efficacy in BRCA-mutated tumors."],
                            ["Aspirin inhibits COX-1 and COX-2 enzymes, reducing prostaglandin synthesis. This mechanism explains its anti-inflammatory effects in rheumatoid arthritis. Drug interactions occur with warfarin, increasing bleeding risk."],
                        ],
                        inputs=text_input,
                        label="📋 Example Texts"
                    )
                    
                    analyze_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    summary_output = gr.Markdown(label="Summary")
                    entity_table = gr.Dataframe(
                        headers=["Entity", "Type"],
                        label="Extracted Entities",
                        wrap=True,
                    )
            
            gr.Markdown("### 📊 Annotated Text")
            highlighted_output = gr.HTML()
            
            gr.Markdown("### 🕸️ Knowledge Graph with Relationships")
            kg_output = gr.HTML()
            
            gr.Markdown("### 📋 Extracted Relationships")
            relations_output = gr.Markdown()
            
            analyze_btn.click(
                fn=process_text,
                inputs=[text_input, extract_rels_cb],
                outputs=[summary_output, highlighted_output, entity_table, kg_output, relations_output]
            )
        
        # Tab 2: PDF Upload
        with gr.TabItem("📄 PDF Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_input = gr.File(
                        label="Upload Research Paper (PDF)",
                        file_types=[".pdf"],
                    )
                    extract_rels_pdf = gr.Checkbox(
                        label="🔗 Extract Relationships",
                        value=True,
                    )
                    analyze_pdf_btn = gr.Button("🔍 Process PDF", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    pdf_summary = gr.Markdown(label="Summary")
                    pdf_entity_table = gr.Dataframe(
                        headers=["Entity", "Type"],
                        label="Extracted Entities",
                        wrap=True,
                    )
            
            extracted_text = gr.Textbox(
                label="📄 Extracted Text Preview",
                lines=5,
                interactive=False,
            )
            
            gr.Markdown("### 📊 Annotated Text")
            pdf_highlighted = gr.HTML()
            
            gr.Markdown("### 🕸️ Knowledge Graph")
            pdf_kg = gr.HTML()
            
            gr.Markdown("### 📋 Extracted Relationships")
            pdf_relations = gr.Markdown()
            
            analyze_pdf_btn.click(
                fn=process_pdf,
                inputs=[pdf_input, extract_rels_pdf],
                outputs=[pdf_summary, pdf_highlighted, pdf_entity_table, extracted_text, pdf_kg, pdf_relations]
            )
    
    gr.Markdown("""
    ---
    ### 🔬 About This Pipeline
    
    - **NER Model**: BioBERT + LoRA fine-tuned on BIORED (85% F1)
    - **RE Model**: BioBERT + LoRA for relation extraction (47% F1)
    - **Knowledge Graph**: NetworkX + Matplotlib visualization
    
    The graph shows **actual predicted relationships** between entities, not just potential ones!
    """)


# Launch
if __name__ == "__main__":
    print("\n🚀 Starting Gradio app on http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
