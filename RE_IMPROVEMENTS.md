# Relation Extraction (RE) Model Improvements

## Changes Made to Improve Performance (from 49.7% to target 55-60% F1)

### 1. **Increased Model Capacity**
- **LoRA Rank**: 32 → **48** (+50% parameters)
- **LoRA Alpha**: 64 → **96**
- More capacity to learn complex relationship patterns

### 2. **Use All Training Data**
- **max_samples_per_class**: 2500 → **None** (unlimited)
- Previously limited to 2500 samples per class
- Now uses **full BIORED dataset** (~30K+ training pairs)
- More data = better generalization

### 3. **Longer Context Window**
- **max_length**: 256 → **384** tokens (+50%)
- Captures more context around entity pairs
- Better understanding of relationship semantics

### 4. **Extended Training**
- **num_epochs**: 5 → **8** (+60% training steps)
- **early_stopping_patience**: 3 → **5**
- More time for model to converge on complex patterns

### 5. **Focal Loss Implementation**
- Added **FocalLoss** class with gamma=2.0
- Focuses training on hard-to-classify examples
- Better handling of class imbalance
- Reduces bias toward majority classes

### 6. **Entity Type Information**
- Enhanced input with entity types: `[E1:Gene]` instead of just `[E1]`
- Model now knows entity types when predicting relations
- Example: `[E1:Chemical] Metformin [/E1] activates [E2:Gene] AMPK [/E2]`
- Helps model learn type-specific relationship patterns

### 7. **Better Training Configuration**
- **Learning Rate**: 2e-4 → **1.5e-4** (more stable)
- **Warmup Ratio**: 0.1 → **0.15** (smoother start)
- **LR Scheduler**: linear → **cosine** (better convergence)
- **Label Smoothing**: 0.0 → **0.1** (regularization)
- **Gradient Accumulation**: 8 → **10** steps

### 8. **Adjusted for Memory Constraints**
- **batch_size**: 4 → **3** (to fit larger context on 4GB GPU)
- **Effective batch size**: 32 → **30** (batch_size × gradient_accumulation)

## Expected Performance Gains

| Improvement | Expected F1 Gain |
|-------------|------------------|
| All training data | +3-5% |
| Larger LoRA rank | +2-3% |
| Entity type info | +2-4% |
| Focal loss | +1-2% |
| Longer context | +1-2% |
| More epochs + better schedule | +1-2% |
| **Total Expected** | **+10-18%** |

**Previous**: 49.7% F1 (weighted)  
**Target**: 55-60% F1 (weighted)  
**Optimistic**: 60-65% F1 (weighted)

## Training Time Estimate

- Previous: ~30 minutes (5 epochs, 2500/class)
- New: **~60-90 minutes** (8 epochs, full data)
  - More epochs: +60%
  - More data: +100-150%
  - Offset by better GPU utilization

## How to Run

```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Train improved RE model
python train_re.py

# Will save to: models/biobert-re-lora/
```

## Technical Details

### Focal Loss Formula
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- γ=2.0: Focusing parameter (reduces loss for well-classified)
- α_t: Class weights (handles imbalance)

### Entity Type Markers
```python
# Before: Generic markers
[E1] Metformin [/E1] activates [E2] AMPK [/E2]

# After: Type-aware markers
[E1:Chemical] Metformin [/E1] activates [E2:Gene] AMPK [/E2]
```

### Training Stats (Estimated)
- **Total Training Samples**: ~30,000 pairs
- **Training Steps**: ~8,000 steps (8 epochs × 1000 steps/epoch)
- **Evaluation**: Every 200 steps
- **GPU Memory**: ~3.8GB / 4GB VRAM
- **Time per Epoch**: ~7-11 minutes

## Monitoring Training

Watch for these metrics during training:
- **Epoch 1-2**: Loss should drop quickly (2.1 → 1.5)
- **Epoch 3-5**: F1 should reach 50-55%
- **Epoch 6-8**: F1 should reach 55-60%+
- **Best checkpoint**: Saved when eval F1 peaks

## Next Steps After Training

1. **Test the model** - Check test set F1 score
2. **Update Gradio app** - May need to handle new special tokens
3. **Compare predictions** - Verify improvements on sample texts
4. **Error analysis** - Check which relation types improved most

## Potential Further Improvements (if needed)

If target not reached, consider:
- **Data augmentation** (synonym replacement, back-translation)
- **Ensemble methods** (train multiple models, average predictions)
- **Post-processing rules** (type-based filtering)
- **BioBERT-Large** (340M params vs 110M)
- **Multi-task learning** (train NER + RE jointly)
