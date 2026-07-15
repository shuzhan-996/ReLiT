# DPT-CMCM: Dynamic Primary-guided Token Adapter

这个版本是在原始 `MSE-Adapter / CMCM` 基础上新增的实验模型，原始 `cmcm` 没有删除，可以继续作为 baseline 使用。

## 新增内容

新增模型文件：

```text
models/multiTask/DPT_CMCM.py
```

新增模型名：

```text
dpt_cmcm
```

核心结构：

1. **Label-aware Dynamic Modality Router**  
   根据单模态预测误差构造软模态目标 `q`，监督路由器输出 `alpha=[alpha_t, alpha_a, alpha_v]`。

2. **Modality-specific Pseudo-token Generation**  
   分别生成 `P_t`, `P_a`, `P_v`，再用 `alpha` 动态组合：

   ```text
   P_dyn = alpha_t * P_t + alpha_a * P_a + alpha_v * P_v
   ```

3. **Primary-guided Token-level Cross-modal Enhancement**  
   使用 `P_dyn` 作为 Query，`[P_t, P_a, P_v]` 作为 Key/Value 做 cross-attention，得到增强后的 pseudo tokens `P_enh`。

最终仍然输入冻结 ChatGLM3：

```text
[P_enh ; Text Tokens ; Task Prompt] -> Frozen ChatGLM3 -> sentiment prediction
```

## 运行方式

原始 baseline：

```bash
python run.py --modelName cmcm --datasetName mosei --train_mode regression --root_dataset_dir D:\\sz\\datasets --pretrain_LM D:\\sz\\models\\chatglm3-6b --gpu_ids 0
```

新模型：

```bash
python run.py --modelName dpt_cmcm --datasetName mosei --train_mode regression --root_dataset_dir D:\\sz\\datasets --pretrain_LM D:\\sz\\models\\chatglm3-6b --gpu_ids 0
```

SIMS-V2：

```bash
python run.py --modelName dpt_cmcm --datasetName simsv2 --train_mode regression --root_dataset_dir D:\\sz\\datasets --pretrain_LM D:\\sz\\models\\chatglm3-6b --gpu_ids 0
```

## 重要超参数

在 `config/config_regression.py` 中已经加入：

```python
'router_hidden_dim': 256,
'router_tau': 0.5,
'router_lambda': 0.05,
'uni_lambda': 0.05,
'enhance_heads': 4,
'adapter_dropout': 0.1,
```

如果训练不稳定，优先把下面两个参数调小：

```python
'router_lambda': 0.01,
'uni_lambda': 0.01,
```

## 建议实验顺序

1. 先跑 `cmcm`，保存原始 MSE-Adapter 结果。
2. 再跑 `dpt_cmcm`，看 loss 是否正常下降。
3. 做消融：
   - 原始 `cmcm`
   - `dpt_cmcm` 去掉 router loss：`router_lambda=0`
   - `dpt_cmcm` 去掉 unimodal loss：`uni_lambda=0`
   - `dpt_cmcm` 完整模型

## 注意

当前压缩包只修改了 `MSE-ChatGLM3-6B` 分支，Qwen 和 LLaMA2 分支未同步修改。建议先在 ChatGLM3 上跑通，再迁移到其他 backbone。
