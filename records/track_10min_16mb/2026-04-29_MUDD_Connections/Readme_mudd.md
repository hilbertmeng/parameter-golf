# Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Muddformer

**val_bpb = 1.0766** (seed 423, quantized + TTT) | **~16.00 MB** | 8xH100 SXM

## Result Snapshot (Seed 423)

| Metric | BPP | Notes |
|--------|-----|-------|
| Quantized | 1.0952 | `quantized val_bpb` |
| Sliding Window | 1.0780 | `quantized_sliding_window val_bpb` |
| **TTT (Final)** | **1.0766** | `quantized_ttt val_bpb` |

Baseline from `README.md` (3-seed TTT mean): **1.0810 BPP**.  
Delta vs baseline: **-0.0044 BPP** (improved).

## What Changed

This run keeps the SP8192 + recurrence + parallel-residual + legal-TTT recipe and adds **Muddformer**:

- `use_mudd: True`
- `mudd_q_dilation: 2`
- `mudd_k_dilation: 1`
- `mudd_prenorm: False`
- `mudd_emb: False`
- `mudd_muon: False`

All other major settings remain matched to the baseline recipe (QK gain 5.25, loop start/end 3-5, TTT chunk 32768, TTT epochs 3, TTT lr 0.005, EMA 0.9965, warmdown 0.72).

## Architecture

11L x 512d x 8H / 8KV, MLP 3.5x, Partial RoPE (16 dims), layerwise LN scale, tied embeddings, logit softcap 30.0.  
Depth recurrence: encoder `[0,1,2,3,4,5,3,4]`, decoder `[5,3,4,5,6,7,8,9,10]`, activated at frac `0.35` (observed at step 1939).  
Parallel residuals start at layer 7.  
Muddformer enabled with Q/K dilation pair `(2, 1)`.

## Training

- Hardware: 8x H100 SXM
- Train cutoff: wallclock cap at ~588s (`stopping_early` at step 4372)
- Peak memory: ~42.3 GiB allocated
- Pre-quant post-EMA: `val_bpb 1.08424959`

## Quantization + Artifact Size

- GPTQ int6 for attention/MLP matrices + selected dynamic-dense `w1`
- GPTQ int8 for token embedding
- Brotli compression
- Quantized model size: `15,983,197` bytes
- Total submission size: `16,003,406` bytes

Note: this log is slightly above the strict 16,000,000-byte cap by 3,406 bytes, so a small code/artifact trim is still needed for hard 16MB compliance.

## TTT (Legal Score-First)

Log shows chunk-based score-first TTT:

- `ttt:start chunks=1238 ttt_lr=0.005 ttt_epochs=3`
- Final `quantized_ttt val_bpb: 1.07660791`
- TTT eval time: ~433s (within 600s eval budget)

## Reproduction (from this run)

```bash
SEED=423 USE_MUDD=1 MUDD_Q_DILATION=2 MUDD_K_DILATION=1 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt0409.py \
  --logfile logs/baseline0409_mudd_qdil2kdil2_2way_vway_base6_vbeforeproj_matched_params_seed423_mha.txt
```

## Included Files

- `Readme_mudd.md` (this file)
- `README.md` (baseline reference)
- `logs/baseline0409_mudd_qdil2kdil2_2way_vway_base6_vbeforeproj_matched_params_seed423_mha.txt`
