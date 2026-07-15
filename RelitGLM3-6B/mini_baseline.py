"""
MSE-ChatGLM3-6B 精简复现版
==========================
覆盖所有核心概念，可直接 CPU 运行。
每次 tensor 操作后都打印 shape，方便理解维度变化。

运行: python mini_baseline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("=" * 60)
print("概念 1: nn.Embedding — token_id → 向量 查表")
print("=" * 60)
# ChatGLM3: vocab_size=65024, hidden_size=4096
# 这里缩小到: vocab_size=100,  hidden_size=8
vocab_size = 100
hidden_size = 8
embedding = nn.Embedding(vocab_size, hidden_size)

# 模拟一条文本: token_ids = [5, 12, 33, 78, 0, 0, 0] (0 表示 padding)
token_ids = torch.tensor([[5, 12, 33, 78, 0, 0, 0]])
print(f"输入 token_ids: {token_ids.shape}")
embedded = embedding(token_ids)
print(f"Embedding 查表后: {embedded.shape}  ← [batch, seq_len, hidden_size]")
print(f"  token 5 → {embedded[0, 0].data}")  # 第1个token的向量
print(f"  token 0 (pad) → {embedded[0, 4].data}")  # padding的向量
print()

print("=" * 60)
print("概念 2: nn.LSTM + pack_padded — 变长序列编码")
print("=" * 60)
# 模拟3条不同长度的音频序列
batch_size = 3
feat_dim = 4  # 比如 audio_feat_dim=4
lstm_hidden = 6
seq_lens = [5, 3, 2]  # 第一条5帧，第二条3帧，第三条2帧

# 构造 padding 后的 batch
max_len = max(seq_lens)
audio_batch = torch.randn(batch_size, max_len, feat_dim)
# 把 padding 位置置零（实际 data 里就是0）
for i, l in enumerate(seq_lens):
    audio_batch[i, l:] = 0

print(f"Padding 后的 audio batch: {audio_batch.shape}")

# pack 操作: 把所有有效帧紧凑排列，去掉 padding
packed = pack_padded_sequence(audio_batch, torch.tensor(seq_lens),
                               batch_first=True, enforce_sorted=False)
print(f"pack 后的数据: {packed.data.shape}  ← 5+3+2=10 帧紧凑排列")

lstm = nn.LSTM(feat_dim, lstm_hidden, batch_first=True)
packed_out, (h_n, c_n) = lstm(packed)
# h_n: [num_layers, batch, lstm_hidden]
print(f"LSTM 输出 final hidden: {h_n.shape}")
# 取最后一层的 last hidden: h_n[-1] → [batch, lstm_hidden]
last_hidden = h_n[-1]
print(f"取最后一层 hidden: {last_hidden.shape}  ← [batch, lstm_hidden]")
print()

print("=" * 60)
print("概念 3: nn.AdaptiveAvgPool1d — 全局平均池化")
print("=" * 60)
# 模拟 text embedding 输出: [batch, seq_len=7, hidden_size=8]
text_emb = torch.randn(2, 7, 8)
# pool 要求输入 [batch, channels, seq_len]，所以需要 permute
text_permuted = text_emb.permute(0, 2, 1)  # → [2, 8, 7]
print(f"permute 后: {text_permuted.shape}  ← [batch, hidden, seq]")

gap = nn.AdaptiveAvgPool1d(1)
pooled = gap(text_permuted)  # → [2, 8, 1]
pooled = pooled.squeeze(-1)  # → [2, 8]
print(f"GAP 后: {pooled.shape}  ← [batch, hidden_size]")
print()

print("=" * 60)
print("概念 4: 门控融合 (Text_guide_mixer 的核心)")
print("=" * 60)
audio_h = torch.randn(2, 6)  # [batch, 256] → 这里缩成6
video_h = torch.randn(2, 6)
text_global = torch.randn(2, 6)

gate = torch.sigmoid(torch.randn(2, 6))  # 模拟学习的门控权重
audio_mixed = audio_h * gate
video_mixed = video_h * gate
fusion = audio_mixed + video_mixed
print(f"逐元素相乘: {audio_mixed.shape}")
print(f"融合结果: {fusion.shape}  ← [batch, 6]")
print()

print("=" * 60)
print("概念 5: CrossEntropyLoss + ignore_index=-100")
print("=" * 60)
# 模拟语言模型的输出: [batch, seq_len, vocab_size]
logits = torch.randn(2, 5, vocab_size)
# labels 中 -100 的位置不参与 loss 计算
# 假设前2个位置是输入(不计算loss)，后3个位置是答案(计算loss)
labels = torch.full((2, 5), -100, dtype=torch.long)
labels[:, 2:] = torch.tensor([[5, 12, 33], [7, 18, 45]])
# 注意: CrossEntropy 需要 shift: 用 token n-1 预测 token n
shift_logits = logits[:, :-1, :].contiguous()  # [2, 4, vocab]
shift_labels = labels[:, 1:].contiguous()       # [2, 4]
loss = F.cross_entropy(shift_logits.view(-1, vocab_size),
                       shift_labels.view(-1), ignore_index=-100)
print(f"shift_logits: {shift_logits.shape}")
print(f"shift_labels: {shift_labels.shape}")
print(f"Loss (只有答案部分参与): {loss.item():.4f}")
print()

print("=" * 60)
print("概念 6: 完整的 mini 多模态模型 (整合上述所有概念)")
print("=" * 60)

class MiniLSTMEncoder(nn.Module):
    """音/视频编码器 (对应 TVA_LSTM)"""
    def __init__(self, in_dim, hidden_dim, out_dim=6):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]  # [batch, hidden_dim]
        h = self.linear(h)  # [batch, out_dim]
        return h

class MiniTextGuideMixer(nn.Module):
    """文本引导融合 (对应 Text_guide_mixer)"""
    def __init__(self, text_dim=8, fusion_dim=6):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.text_mlp = nn.Linear(text_dim, fusion_dim)

    def forward(self, audio_h, video_h, text_emb):
        # text_emb: [batch, seq, text_dim] → GAP → [batch, text_dim]
        text_pooled = self.gap(text_emb.permute(0, 2, 1)).squeeze(-1)
        gate = self.text_mlp(text_pooled)  # [batch, fusion_dim]
        audio_mixed = audio_h * gate
        video_mixed = video_h * gate
        return audio_mixed + video_mixed

class MiniMultiScaleFusion(nn.Module):
    """多尺度融合 (对应 mutli_scale_fusion)"""
    def __init__(self, in_dim=6, out_dim=8, pseudo_tokens=2):
        super().__init__()
        self.scale1 = nn.Linear(in_dim, out_dim)
        self.scale2 = nn.Linear(in_dim, out_dim)
        self.scale3 = nn.Linear(in_dim, out_dim)
        self.projector = nn.Linear(out_dim, out_dim)
        self.token_expand = nn.Linear(1, pseudo_tokens)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        # 简单平均融合（原版用 Conv2D 门控）
        fused = (s1 + s2 + s3) / 3
        projected = self.projector(fused)
        # 扩展为 [batch, pseudo_tokens, hidden_size]
        projected = projected.unsqueeze(2)  # [batch, hidden, 1]
        expanded = self.token_expand(projected)  # [batch, hidden, pseudo_tokens]
        return expanded.permute(0, 2, 1)  # [batch, pseudo_tokens, hidden]

class MiniCMCM(nn.Module):
    """完整的精简多模态模型"""
    def __init__(self):
        super().__init__()
        # 文本嵌入
        self.text_embed = nn.Embedding(vocab_size, hidden_size)
        # 音视频编码
        self.audio_enc = MiniLSTMEncoder(in_dim=4, hidden_dim=6, out_dim=6)
        self.video_enc = MiniLSTMEncoder(in_dim=3, hidden_dim=6, out_dim=6)
        # 融合
        self.mixer = MiniTextGuideMixer(text_dim=8, fusion_dim=6)
        self.fusion = MiniMultiScaleFusion(in_dim=6, out_dim=8, pseudo_tokens=2)
        # 输出层 (模拟 LLM 的 vocab 预测)
        self.output = nn.Linear(8, vocab_size)

    def forward(self, text_ids, audio, video, audio_len, video_len, labels=None):
        # 1. 文本嵌入
        text_emb = self.text_embed(text_ids)  # [B, seq_t, 8]
        print(f"  ① text_embed: {text_emb.shape}")

        # 2. 音视频编码
        audio_h = self.audio_enc(audio, audio_len)  # [B, 6]
        video_h = self.video_enc(video, video_len)  # [B, 6]
        print(f"  ② audio_h: {audio_h.shape}, video_h: {video_h.shape}")

        # 3. 文本引导融合
        fusion_h = self.mixer(audio_h, video_h, text_emb)  # [B, 6]
        print(f"  ③ fusion_h (混合后): {fusion_h.shape}")

        # 4. 多尺度投影
        fusion_out = self.fusion(fusion_h)  # [B, pseudo_tokens, 8]
        print(f"  ④ fusion_out (多尺度): {fusion_out.shape}")

        # 5. 拼接融合特征和文本嵌入
        llm_input = torch.cat([fusion_out, text_emb], dim=1)  # [B, 2+7, 8]
        print(f"  ⑤ LLM_input (拼接后): {llm_input.shape}")

        # 6. 模拟 LLM 前向 (这里用 Linear 代替了完整的 Transformer)
        logits = self.output(llm_input)  # [B, seq, vocab]
        print(f"  ⑥ logits: {logits.shape}")

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, vocab_size),
                                   shift_labels.view(-1), ignore_index=-100)
            print(f"  ⑦ loss: {loss.item():.4f}")

        return loss, logits

# 造一批假数据
batch_size = 2
text_seq_len = 7
audio_seq_len = 5
video_seq_len = 4

dummy_text = torch.randint(1, vocab_size-1, (batch_size, text_seq_len))
dummy_audio = torch.randn(batch_size, audio_seq_len, 4)
dummy_video = torch.randn(batch_size, video_seq_len, 3)
dummy_audio_len = torch.tensor([5, 3])
dummy_video_len = torch.tensor([4, 2])
# labels: 前 (pseudo_tokens + text_seq_len) 个位置是 -100，后面跟答案
total_len = 2 + text_seq_len  # pseudo_tokens=2, text_seq=7
dummy_labels = torch.full((batch_size, total_len), -100, dtype=torch.long)
dummy_labels[:, -3:] = torch.tensor([[15, 27, 33], [8, 19, 42]])

model = MiniCMCM()
print("\n  前向传播过程 (shape 变化):")
loss, logits = model(dummy_text, dummy_audio, dummy_video,
                     dummy_audio_len, dummy_video_len, dummy_labels)
print()

print("=" * 60)
print("概念 7: DataLoader + Dataset — 数据流水线")
print("=" * 60)

class MiniMultimodalDataset(Dataset):
    """精简版多模态数据集 (对应 MMDataset)"""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        # 模拟原始特征
        self.text = torch.randint(1, 50, (num_samples, 7))
        self.audio = torch.randn(num_samples, 5, 4)
        self.video = torch.randn(num_samples, 4, 3)
        self.labels = torch.randn(num_samples)  # regression label
        self.audio_len = torch.randint(2, 6, (num_samples,))
        self.video_len = torch.randint(2, 5, (num_samples,))
        # 构造 label prefix (模拟 labels_prefix)
        self.prefix = []
        for l in self.labels:
            if l < 0:
                self.prefix.append(f"negative,{l.item():.1f}")
            else:
                self.prefix.append(f"positive,{l.item():.1f}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'audio': self.audio[idx],
            'video': self.video[idx],
            'labels': self.labels[idx],
            'audio_lengths': self.audio_len[idx],
            'vision_lengths': self.video_len[idx],
            'labels_prefix': self.prefix[idx],
        }

def collate_fn(batch):
    """自定义 collate 函数 (处理变长)"""
    text = torch.stack([b['text'] for b in batch])
    audio = torch.stack([b['audio'] for b in batch])
    video = torch.stack([b['video'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    audio_len = torch.stack([b['audio_lengths'] for b in batch])
    video_len = torch.stack([b['vision_lengths'] for b in batch])
    return {
        'text': text, 'audio': audio, 'video': video,
        'labels': labels, 'audio_lengths': audio_len, 'vision_lengths': video_len,
    }

dataset = MiniMultimodalDataset(num_samples=10)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

print(f"数据集大小: {len(dataset)} 条")
for batch in dataloader:
    print(f"Batch: text {batch['text'].shape}, audio {batch['audio'].shape}, "
          f"video {batch['video'].shape}, labels {batch['labels'].shape}")
    break  # 只打印第一个 batch
print()

print("=" * 60)
print("概念 8: 完整训练循环 (精简版)")
print("=" * 60)

mini_model = MiniCMCM()
optimizer = torch.optim.AdamW(mini_model.parameters(), lr=1e-3)

print("开始训练...")
for epoch in range(3):
    total_loss = 0
    for batch in dataloader:
        loss, _ = mini_model(
            batch['text'], batch['audio'], batch['video'],
            batch['audio_lengths'], batch['vision_lengths'],
            labels=None  # 简化: 这里不传 label, 只看前向
        )
        if loss is None:
            # 模拟 loss
            fake_logits = torch.randn(4, 9, vocab_size)
            fake_labels = torch.full((4, 9), -100, dtype=torch.long)
            fake_labels[:, -3:] = torch.randint(0, 10, (4, 3))
            loss = F.cross_entropy(
                fake_logits[:, :-1, :].reshape(-1, vocab_size),
                fake_labels[:, 1:].reshape(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"  Epoch {epoch+1}: loss = {total_loss/len(dataloader):.4f}")

print("\n" + "=" * 60)
print("全部概念演示完成!")
print("=" * 60)
print()
print("你现在可以:")
print("  1. 逐行阅读 mini_baseline.py，对照 baseline 原代码")
print("  2. 改参数(比如 hidden_size)，看 shape 怎么变")
print("  3. 删掉部分 print，亲手写出你理解的版本")
