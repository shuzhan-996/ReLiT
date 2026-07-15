# DPT-DLF-CMCM: DPT Adapter with DLF-style Disentanglement

This branch keeps the original MSE-Adapter code as the baseline (`cmcm`) and keeps the previous dynamic pseudo-token model (`dpt_cmcm`).

The new model is:

```bash
--modelName dpt_dlf_cmcm
```

## What was added

`models/multiTask/DPT_DLF_CMCM.py` adds a DLF-style disentanglement block before dynamic pseudo-token generation.

Pipeline:

```text
Text / Audio / Vision
  -> original encoders
  -> DLF-style shared/private disentanglement
  -> label-aware dynamic router
  -> modality-specific pseudo-token generation
  -> dynamic token routing
  -> primary-guided token-level cross-modal enhancement
  -> Frozen ChatGLM3
```

## Disentanglement details

For each modality vector `h_m`, the module learns:

```text
sh_m = SharedEncoder_m(h_m)
sp_m = PrivateEncoder_m(h_m)
```

Then it reconstructs the compact feature and applies four DLF-style losses:

```text
L_rec     : reconstruct h_m from [sh_m, sp_m]
L_spec    : preserve modality-specific information after reconstruction
L_metric  : align cross-modal shared features and separate batch-shifted negatives
L_orth    : soft orthogonality between shared and private spaces
```

The disentangled representation used by the router/token generators is:

```text
sh_fused = Fuse(sh_t, sh_a, sh_v)
z_m = Combine(sh_fused, sp_m)
```

Final objective:

```text
L = L_llm + router_lambda * L_router + uni_lambda * L_uni + disentangle_lambda * L_disentangle
```

## Run commands

Baseline:

```bash
python run.py --modelName cmcm --datasetName mosei --train_mode regression --root_dataset_dir D:\sz\datasets --pretrain_LM D:\sz\models\chatglm3-6b --gpu_ids 0
```

DPT without disentanglement:

```bash
python run.py --modelName dpt_cmcm --datasetName mosei --train_mode regression --root_dataset_dir D:\sz\datasets --pretrain_LM D:\sz\models\chatglm3-6b --gpu_ids 0
```

DPT + DLF-style disentanglement:

```bash
python run.py --modelName dpt_dlf_cmcm --datasetName mosei --train_mode regression --root_dataset_dir D:\sz\datasets --pretrain_LM D:\sz\models\chatglm3-6b --gpu_ids 0
```

## Recommended first settings

The config already includes conservative defaults:

```python
disentangle_lambda = 0.03
disentangle_rec_weight = 1.0
disentangle_spec_weight = 1.0
disentangle_metric_weight = 0.1
disentangle_orth_weight = 0.1
```

If the loss is unstable, first reduce:

```python
disentangle_lambda = 0.01
```

If training is stable but the improvement is small, try:

```python
disentangle_lambda = 0.05
```
