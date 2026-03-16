# Deep Dive: 4-Bit Quantization (NF4)

A comprehensive technical guide to understanding how 32-bit floating-point weights are compressed to 4-bit and reconstructed during neural network computation.

---

## Part 1: Understanding Float32 Representation

### Standard 32-bit Float (FP32)
```
32 bits total = 1 sign bit + 8 exponent bits + 23 mantissa bits

Example: 0.15625
Binary: 0 01111100 01000000000000000000000
        ↑    ↑            ↑
      sign exponent   mantissa

Memory per weight: 4 bytes
Range: ±3.4 × 10³⁸
Precision: ~7 decimal digits
```

**For a 110M parameter model (BioBERT):**
```
110,000,000 params × 4 bytes = 440 MB
```

---

## Part 2: Naive 4-Bit Quantization (Simple Approach)

### Linear Quantization (Not Used - Just for Understanding)

```python
# Step 1: Find the range of weights
weights = [-0.8, -0.3, 0.0, 0.2, 0.5, 0.9]
min_w = -0.8
max_w = 0.9
range_w = 1.7

# Step 2: Map to 4-bit integers (0-15)
# 4 bits = 2^4 = 16 possible values

scale = (max_w - min_w) / 15  # = 1.7 / 15 = 0.1133

# Step 3: Quantize each weight
quantized = []
for w in weights:
    q = round((w - min_w) / scale)  # Maps to 0..15
    quantized.append(q)

# Results:
# -0.8 → 0
# -0.3 → 4
#  0.0 → 7
#  0.2 → 9
#  0.5 → 11
#  0.9 → 15

# Step 4: Dequantize (reconstruct approximate value)
dequantized = []
for q in quantized:
    w_approx = min_w + (q * scale)
    dequantized.append(w_approx)

# Reconstructed:
# 0 → -0.8000 ✓
# 4 → -0.3467 (error: 0.0467)
# 7 → -0.0066 (error: 0.0066)
# 9 →  0.2200 (error: 0.0200)
```

**Problem:** Uniform distribution wastes precision where weights are sparse!

---

## Part 3: NF4 Quantization (What Your Code Uses)

### The Key Insight

Neural network weights follow a **normal distribution** centered near zero:

```
    Many weights here ↓
         ████████
        ██████████
       ████████████
     ██████████████
    ████████████████
─────────0─────────────→
      Few weights here
```

**NF4 Strategy:** Allocate more quantization levels near zero, fewer at extremes.

### NF4 Quantization Levels

Instead of uniform spacing (0, 1, 2, ..., 15), NF4 uses **quantiles of standard normal distribution**:

```python
# NF4 lookup table (16 values for 4 bits)
NF4_VALUES = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

# Note: More levels between -0.3 and +0.3 (where most weights are)
```

Visual representation:
```
-1.0 -------- -0.7 -- -0.5 -- -0.4 - -0.28 - -0.18 - -0.09 - 0.0 - 0.08 - 0.16 - 0.25 - 0.34 - 0.44 -- 0.56 ---- 0.72 -------- 1.0
  0            1      2       3      4       5        6      7     8      9      10     11     12      13       14           15

Dense spacing here → [--------] ← Captures most weights accurately
```

---

## Part 4: The Complete Quantization Pipeline

### Step-by-Step Process

#### **1. Block-wise Quantization**

Don't quantize all weights together - do it in small blocks (typically 64 or 128 weights at a time):

```python
# Example: Weight matrix of shape [768, 768]
# Split into blocks of 64 weights each

Block 1: weights[0:64]     → has its own scale factor
Block 2: weights[64:128]   → has its own scale factor
Block 3: weights[128:192]  → has its own scale factor
...
```

**Why blocks?** Different layers/regions have different ranges. Per-block scaling preserves precision.

#### **2. Compute Scale Factor (Absmax)**

For each block:

```python
block = [-0.3, 0.15, -0.08, 0.42, 0.19, ...]  # 64 weights

# Find absolute maximum
absmax = max(abs(block))  # = 0.42

# Scale factor to normalize to [-1, 1]
scale = absmax / 1.0  # = 0.42

# Normalize
normalized = [w / scale for w in block]
# [-0.714, 0.357, -0.190, 1.0, 0.452, ...]
```

#### **3. Map to Nearest NF4 Value**

```python
def quantize_to_nf4(normalized_weight):
    """Find closest NF4 value."""
    min_error = float('inf')
    best_idx = 0
    
    for idx, nf4_val in enumerate(NF4_VALUES):
        error = abs(normalized_weight - nf4_val)
        if error < min_error:
            min_error = error
            best_idx = idx
    
    return best_idx  # 4-bit integer (0-15)

# Example:
normalize_weight = 0.357
# Closest NF4 values: 0.3379 (idx=11) or 0.4407 (idx=12)
# Distance to 0.3379: |0.357 - 0.3379| = 0.0191
# Distance to 0.4407: |0.357 - 0.4407| = 0.0837
# → Choose idx=11
```

#### **4. Store Quantized Weights + Metadata**

```python
# For each block (64 weights):
quantized_block = [7, 11, 6, 15, 12, ...]   # 64 × 4 bits = 256 bits = 32 bytes
scale_factor = 0.42                          # 32-bit float = 4 bytes

# Total per block: 32 + 4 = 36 bytes
# vs Original: 64 × 4 = 256 bytes
# Compression: 256/36 = 7.1x
```

---

## Part 5: Dequantization (Reconstruction)

When computing forward pass:

```python
def dequantize_nf4(quantized_idx, scale):
    """Reconstruct approximate weight."""
    nf4_value = NF4_VALUES[quantized_idx]
    original_weight ≈ nf4_value * scale
    return original_weight

# Example:
quantized_idx = 11  # From quantization step
scale = 0.42
reconstructed = NF4_VALUES[11] * 0.42
              = 0.3379 * 0.42
              = 0.1419

# Original weight: 0.15
# Reconstructed:   0.1419
# Error:           0.0081 (5.4% relative error)
```

---

## Part 6: Double Quantization (The "Double" in BitsAndBytes)

Your config uses this:
```python
bnb_4bit_use_double_quant=True
```

**Problem:** We still need to store scale factors in FP32 (4 bytes each)!

**Solution:** Quantize the scale factors too! 🤯

```python
# Original approach:
Block 1: 64 weights (4-bit) + scale (32-bit) = 32 + 4 = 36 bytes
Block 2: 64 weights (4-bit) + scale (32-bit) = 32 + 4 = 36 bytes
...
10,000 blocks: scales = 10,000 × 4 bytes = 40 KB

# Double quantization:
# Group scales into super-blocks and quantize them to FP8:
scales = [0.42, 0.38, 0.51, ..., 0.29]  # 10,000 scales
super_block_scale = max(scales) / 127  # = 0.51 / 127 = 0.004

quantized_scales = [round(s / super_block_scale) for s in scales]
# Each scale now uses 8 bits instead of 32 bits

# Storage:
10,000 × 1 byte (FP8) + few super-block scales (32-bit)
= 10 KB + ~40 bytes
= ~10 KB (4x savings on scales!)
```

---

## Part 7: The Complete BitsAndBytes Config

```python
BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",              # Use NF4 (normal float 4-bit)
    bnb_4bit_use_double_quant=True,         # Quantize scale factors too
    bnb_4bit_compute_dtype=torch.float16,   # Dequantize to FP16 for computation
)
```

### What Happens During Forward Pass:

```python
# 1. Input arrives
input = [0.2, -0.1, 0.5, ...]  # FP16

# 2. Dequantize weights on-the-fly (per block)
for block in quantized_model:
    # Load 4-bit weights + scale
    scale = load_scale(block_id)
    nf4_indices = load_quantized_weights(block_id)  # [7, 11, 6, ...]
    
    # Dequantize to FP16
    weights_fp16 = [NF4_VALUES[idx] * scale for idx in nf4_indices]
    
    # Compute in FP16
    output = matmul(input, weights_fp16)  # Fast GPU FP16 ops
    
    # Weights stay in 4-bit in memory!

# 3. Return output
```

**Key:** Weights remain in 4-bit in VRAM. Only temporary FP16 copies exist during computation.

---

## Part 8: Memory Comparison

### BioBERT (110M parameters)

| Format | Bits per Weight | Total Size | VRAM on RTX 3050 |
|--------|----------------|------------|------------------|
| FP32 | 32 | 110M × 4B = 440 MB | ❌ Too large with activations |
| FP16 | 16 | 110M × 2B = 220 MB | ⚠️ Tight fit |
| **4-bit (NF4)** | 4 | 110M × 0.5B = **55 MB** | ✅ Fits easily! |
| **+ Double Quant** | ~4 | **~56 MB** | ✅ Best option |

**Total VRAM Usage During Training:**
```
4-bit NF4:
- Model weights: 56 MB
- LoRA adapters (FP16): ~20 MB
- Activations: ~1.5 GB (batch_size=6, seq_len=240)
- Optimizer states: ~40 MB (only for LoRA params)
- Gradients: ~40 MB
────────────────────────────────
Total: ~1.66 GB / 4 GB available ✅

Without Quantization (FP16):
- Model weights: 220 MB
- LoRA adapters: 20 MB
- Activations: 1.5 GB
- Optimizer: 500 MB (for all params)
- Gradients: 240 MB
────────────────────────────────
Total: ~2.48 GB + overhead ≈ 3.2 GB ⚠️ Very tight!
```

---

## Part 9: Quantization Error Analysis

### Error at Different Precision Levels

```python
Original weight: 0.15625

FP32: 0.15625000000000000 (exact)
FP16: 0.15625000000000000 (exact for this value)
NF4:  0.16093020141124725 × 0.42 ≈ 0.1610
      Error: 0.0048 (3.0%)

# For weight = -0.3:
FP32: -0.30000000
NF4:  -0.28444138 × 1.0 ≈ -0.2844
      Error: 0.0156 (5.2%)
```

**Why Models Still Work:**

1. **Aggregation Effect:** Errors average out across thousands of weights
2. **LoRA Adapters:** Trainable FP16 adapters compensate for quantization errors
3. **Fine-tuning:** Model learns to work with quantized weights
4. **Redundancy:** Neural networks are over-parameterized

### Theoretical Precision:

```
4 bits = 16 discrete levels
Dynamic range: ~[-1, 1] per block
Precision: ~0.125 increments (but non-uniform with NF4)
Signal-to-Noise Ratio: ~24 dB
```

---

## Part 10: Why NF4 > Other Methods

| Method | Distribution | Precision Center | Overhead |
|--------|-------------|------------------|----------|
| **Int4 Uniform** | Equal spacing | Poor at zero | Low |
| **Int4 Asymmetric** | Custom range | Medium | Medium |
| **NF4** | Normal distribution | ✅ Excellent at zero | Low |
| **GPTQ** | Data-calibrated | Very good | High (calibration) |

**NF4 wins for transformers** because attention weights naturally follow normal distribution.

---

## Visualization: The Full Pipeline

```
┌─────────────────┐
│ FP32 Weights    │  Original: [-0.3, 0.15, -0.08, 0.42, ...]
│ [110M params]   │  Size: 440 MB
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ 1. Block Division (64/block)│
│    Block 1: [-0.3, 0.15,...]│
│    Scale: absmax = 0.42     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 2. Normalize to [-1, 1]     │
│    [-0.714, 0.357, ...]     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 3. Map to NF4 (0-15)        │
│    [3, 11, 6, ...]          │  Now 4-bit per weight!
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ 4. Store with Scale         │
│    Weights: 32 bytes        │
│    Scale: 4 bytes           │
└────────┬────────────────────┘
         │
         ▼  (Double Quantization)
┌─────────────────────────────┐
│ 5. Quantize Scales (FP8)    │
│    Scales: 1 byte each      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│ Final: 55 MB    │  8x compression!
│ [110M params]   │
└─────────────────┘

═══════════════════════════════════════
         FORWARD PASS
═══════════════════════════════════════

┌─────────────────┐
│ Input (FP16)    │  [batch, seq_len, hidden]
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Dequantize Block (on GPU)   │
│   4-bit [7,11,6] → FP16     │
│   [0.0, 0.142, -0.076, ...] │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Matrix Multiply (FP16)      │
│   output = input @ weights  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│ Output (FP16)   │
└─────────────────┘
```

---

## Why Use 4-bit Quantization Only for RE?

### Comparison: NER vs RE

| Aspect | **NER (No Quantization)** | **RE (4-bit Quantization)** |
|--------|---------------------------|----------------------------|
| **Task Type** | Token classification (per-token precision needed) | Sequence classification (single output) |
| **Memory Needs** | Lower (batch_size=8, simpler task) | Higher (needs more VRAM) |
| **Data Size** | Smaller, balanced | Larger, highly imbalanced |
| **Tolerance** | Less tolerant to precision loss | More tolerant to quantization |
| **VRAM Usage** | ~2-3GB (fits with FP16) | Would be ~6GB without quantization |

**Key Reasons:**

1. **NER needs per-token precision**: Every token gets a label (B-Disease, I-Gene, etc.). Quantization can blur boundaries between tokens, hurting IOB tagging accuracy.

2. **RE is sequence-level**: One classification per entity pair. Quantization error is less critical since there's aggregation over the whole sequence.

3. **VRAM constraints**: RTX 3050 has only 4GB. RE with more complex examples (240 tokens, batch_size=6, gradient_accumulation) needs aggressive memory optimization.

---

## Summary

**The Magic:**
1. **32-bit → 4-bit:** Block-wise normalization + NF4 lookup table
2. **Storage:** 55 MB instead of 440 MB (8x savings)
3. **Computation:** Dequantize just-in-time to FP16 for fast GPU ops
4. **Accuracy:** ~3-5% error per weight, but aggregates well in networks
5. **Double Quant:** Even scales get compressed (FP32 → FP8)

**Why It Works:**
- Neural network weights are normally distributed → NF4 is optimal
- LoRA adapters in FP16 compensate for quantization errors
- Redundant parameters → some precision loss is acceptable
- Enables training on 4GB consumer GPUs! 🚀

**Trade-offs:**
- Small accuracy loss (~1-2% on benchmarks)
- Slight computation overhead (dequantization)
- Cannot train base model weights (frozen at 4-bit)
- Perfect for fine-tuning with LoRA!

---

## References

- **BitsAndBytes Paper**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **NF4**: Based on quantiles of standard normal distribution N(0,1)
- **Implementation**: https://github.com/TimDettmers/bitsandbytes
