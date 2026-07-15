# Dynamic Primary-guided Token Adapter with DLF-style Disentanglement for MSE-Adapter (ChatGLM3-6B)
# This file keeps the original MSE-Adapter/Frozen-LLM training pipeline,
# and replaces the original TGM+MSF pseudo-token generator with:
#   1) label-aware dynamic modality routing,
#   2) modality-specific pseudo-token generation,
#   3) primary-guided token-level cross-modal enhancement.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from models.subNets.Textmodel import Language_model

__all__ = ['DPT_DLF_CMCM']


class DPT_DLF_CMCM(nn.Module):
    """MSE-Adapter variant with dynamic pseudo-token routing.

    Expected usage:
        python run.py --modelName dpt_cmcm --datasetName mosei ...

    The frozen ChatGLM3 Language_model is kept unchanged. Only the lightweight
    adapter part is trained.
    """

    def __init__(self, args):
        super(DPT_DLF_CMCM, self).__init__()
        self.args = args

        # Frozen LLM and text embedding from original MSE-Adapter
        self.LLM = Language_model(args)

        text_in, audio_in, video_in = args.feature_dims[:]

        # Keep the original audio/video LSTM encoders to minimize code risk.
        self.audio_LSTM = TVA_LSTM(
            audio_in,
            args.a_lstm_hidden_size,
            num_layers=args.a_lstm_layers,
            dropout=args.a_lstm_dropout,
        )
        self.video_LSTM = TVA_LSTM(
            video_in,
            args.v_lstm_hidden_size,
            num_layers=args.v_lstm_layers,
            dropout=args.v_lstm_dropout,
        )

        hidden_dim = getattr(args, 'router_hidden_dim', 256)
        pseudo_tokens = getattr(args, 'pseudo_tokens', 4)
        enhance_heads = getattr(args, 'enhance_heads', 4)
        router_tau = getattr(args, 'router_tau', 0.5)
        dropout = getattr(args, 'adapter_dropout', 0.1)

        self.dynamic_token_adapter = DynamicTokenAdapter(
            text_dim=text_in,
            hidden_dim=hidden_dim,
            pseudo_tokens=pseudo_tokens,
            num_heads=enhance_heads,
            tau=router_tau,
            dropout=dropout,
            disentangle_margin=getattr(args, 'disentangle_margin', 0.2),
            loss_rec_weight=getattr(args, 'disentangle_rec_weight', 1.0),
            loss_spec_weight=getattr(args, 'disentangle_spec_weight', 1.0),
            loss_metric_weight=getattr(args, 'disentangle_metric_weight', 0.1),
            loss_orth_weight=getattr(args, 'disentangle_orth_weight', 0.1),
        )

        self.router_lambda = getattr(args, 'router_lambda', 0.05)
        self.uni_lambda = getattr(args, 'uni_lambda', 0.05)
        self.disentangle_lambda = getattr(args, 'disentangle_lambda', 0.05)

    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        # text: [B, raw_seq, ...] -> ChatGLM input embeddings [B, L, 4096]
        text = self.LLM.text_embedding(text[:, 0, :].long())

        audio_h = self.audio_LSTM(audio, audio_len)  # [B, 256]
        video_h = self.video_LSTM(video, video_len)  # [B, 256]

        adapter_out = self.dynamic_token_adapter(
            text_embed=text,
            audio_h=audio_h,
            video_h=video_h,
            labels=labels,
        )
        fusion_h = adapter_out['pseudo_tokens']  # [B, pseudo_tokens, text_dim]

        LLM_input = torch.cat([fusion_h, text], dim=1)
        LLM_output = self.LLM(LLM_input, labels)

        loss = LLM_output.loss
        if 'router_loss' in adapter_out:
            loss = loss + self.router_lambda * adapter_out['router_loss']
        if 'uni_loss' in adapter_out:
            loss = loss + self.uni_lambda * adapter_out['uni_loss']
        if 'disentangle_loss' in adapter_out:
            loss = loss + self.disentangle_lambda * adapter_out['disentangle_loss']

        res = {
            'Loss': loss,
            'Loss_llm': LLM_output.loss.detach(),
            'Loss_router': adapter_out.get('router_loss', torch.tensor(0.0, device=loss.device)).detach(),
            'Loss_uni': adapter_out.get('uni_loss', torch.tensor(0.0, device=loss.device)).detach(),
            'Loss_disentangle': adapter_out.get('disentangle_loss', torch.tensor(0.0, device=loss.device)).detach(),
            'Alpha': adapter_out['alpha'].detach(),
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        text = self.LLM.text_embedding(text[:, 0, :].long())
        audio_h = self.audio_LSTM(audio, audio_len)
        video_h = self.video_LSTM(video, video_len)

        adapter_out = self.dynamic_token_adapter(
            text_embed=text,
            audio_h=audio_h,
            video_h=video_h,
            labels=None,
        )
        fusion_h = adapter_out['pseudo_tokens']
        LLM_input = torch.cat([fusion_h, text], dim=1)
        LLM_output = self.LLM.generate(LLM_input)
        return LLM_output


class TVA_LSTM(nn.Module):
    """Original MSE-Adapter audio/video LSTM encoder."""

    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(TVA_LSTM, self).__init__()
        self.rnn = nn.LSTM(
            in_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 256)

    def forward(self, x, lengths):
        packed_sequence = pack_padded_sequence(
            x,
            lengths.to('cpu'),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = self.linear(h)
        return h




class DLFDisentangler(nn.Module):
    """DLF-style shared/private disentanglement block for vector features.

    The original DLF paper disentangles each modality into modality-shared and
    modality-specific subspaces and regularizes them with reconstruction,
    specific consistency, metric/alignment, and orthogonality constraints.
    This implementation adapts that idea to the MSE-Adapter/Frozen-LLM pipeline:
    it operates on the compact modality vectors h_t, h_a, h_v before pseudo-token
    generation.
    """

    def __init__(self, hidden_dim=256, dropout=0.1, margin=0.2,
                 loss_rec_weight=1.0, loss_spec_weight=1.0,
                 loss_metric_weight=0.1, loss_orth_weight=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.loss_rec_weight = loss_rec_weight
        self.loss_spec_weight = loss_spec_weight
        self.loss_metric_weight = loss_metric_weight
        self.loss_orth_weight = loss_orth_weight

        def mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

        # DLF-style modality-shared and modality-specific encoders.
        # We keep independent shared encoders for each modality because their
        # input distributions differ, then constrain the shared outputs to align.
        self.shared_t = mlp(hidden_dim, hidden_dim)
        self.shared_a = mlp(hidden_dim, hidden_dim)
        self.shared_v = mlp(hidden_dim, hidden_dim)

        self.private_t = mlp(hidden_dim, hidden_dim)
        self.private_a = mlp(hidden_dim, hidden_dim)
        self.private_v = mlp(hidden_dim, hidden_dim)

        # Modality decoders reconstruct the original compact features from
        # [shared, private], following DLF's reconstruction regularization.
        self.decoder_t = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder_v = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fuse the aligned shared representations into one shared semantic anchor.
        self.shared_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        # For each modality, combine global shared affective semantics with its
        # own modality-specific cues before pseudo-token generation.
        self.combine_t = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.combine_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.combine_v = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _cosine_distance(x, y):
        return 1.0 - F.cosine_similarity(x, y, dim=-1)

    @staticmethod
    def _orthogonality_loss(shared, private):
        # Squared cosine makes positive and negative correlation both penalized.
        cos = F.cosine_similarity(shared, private, dim=-1)
        return torch.mean(cos ** 2)

    def _metric_loss(self, sh_t, sh_a, sh_v):
        # Positive: same sample across modalities should be close in shared space.
        pos = (
            self._cosine_distance(sh_t, sh_a).mean() +
            self._cosine_distance(sh_t, sh_v).mean() +
            self._cosine_distance(sh_a, sh_v).mean()
        ) / 3.0

        # Negative: a simple batch-shifted contrastive term. This is a lightweight
        # adaptation of DLF's metric regularization for regression batches.
        if sh_t.size(0) <= 1:
            return pos
        sh_t_neg = torch.roll(sh_t, shifts=1, dims=0)
        sh_a_neg = torch.roll(sh_a, shifts=1, dims=0)
        sh_v_neg = torch.roll(sh_v, shifts=1, dims=0)
        neg_sim = (
            F.cosine_similarity(sh_t, sh_t_neg, dim=-1).mean() +
            F.cosine_similarity(sh_a, sh_a_neg, dim=-1).mean() +
            F.cosine_similarity(sh_v, sh_v_neg, dim=-1).mean()
        ) / 3.0
        neg = F.relu(neg_sim - self.margin)
        return pos + neg

    def forward(self, h_t, h_a, h_v):
        sh_t = self.shared_t(h_t)
        sh_a = self.shared_a(h_a)
        sh_v = self.shared_v(h_v)

        sp_t = self.private_t(h_t)
        sp_a = self.private_a(h_a)
        sp_v = self.private_v(h_v)

        rec_t = self.decoder_t(torch.cat([sh_t, sp_t], dim=-1))
        rec_a = self.decoder_a(torch.cat([sh_a, sp_a], dim=-1))
        rec_v = self.decoder_v(torch.cat([sh_v, sp_v], dim=-1))

        # DLF reconstruction loss L_r.
        rec_loss = (
            F.mse_loss(rec_t, h_t) +
            F.mse_loss(rec_a, h_a) +
            F.mse_loss(rec_v, h_v)
        ) / 3.0

        # DLF specific consistency loss L_s: reconstructed features should preserve
        # modality-specific information under the same private encoder.
        sp_t_rec = self.private_t(rec_t)
        sp_a_rec = self.private_a(rec_a)
        sp_v_rec = self.private_v(rec_v)
        spec_loss = (
            F.mse_loss(sp_t_rec, sp_t.detach()) +
            F.mse_loss(sp_a_rec, sp_a.detach()) +
            F.mse_loss(sp_v_rec, sp_v.detach())
        ) / 3.0

        # DLF metric/alignment loss L_m for the shared space.
        metric_loss = self._metric_loss(sh_t, sh_a, sh_v)

        # DLF soft orthogonality loss L_o between shared and private spaces.
        orth_loss = (
            self._orthogonality_loss(sh_t, sp_t) +
            self._orthogonality_loss(sh_a, sp_a) +
            self._orthogonality_loss(sh_v, sp_v)
        ) / 3.0

        disentangle_loss = (
            self.loss_rec_weight * rec_loss +
            self.loss_spec_weight * spec_loss +
            self.loss_metric_weight * metric_loss +
            self.loss_orth_weight * orth_loss
        )

        sh_fused = self.shared_fusion(torch.cat([sh_t, sh_a, sh_v], dim=-1))
        z_t = self.combine_t(torch.cat([sh_fused, sp_t], dim=-1))
        z_a = self.combine_a(torch.cat([sh_fused, sp_a], dim=-1))
        z_v = self.combine_v(torch.cat([sh_fused, sp_v], dim=-1))

        return {
            'z_t': z_t,
            'z_a': z_a,
            'z_v': z_v,
            'sh_t': sh_t,
            'sh_a': sh_a,
            'sh_v': sh_v,
            'sp_t': sp_t,
            'sp_a': sp_a,
            'sp_v': sp_v,
            'sh_fused': sh_fused,
            'rec_loss': rec_loss,
            'spec_loss': spec_loss,
            'metric_loss': metric_loss,
            'orth_loss': orth_loss,
            'disentangle_loss': disentangle_loss,
        }


class DynamicTokenAdapter(nn.Module):
    """Dynamic primary-guided pseudo-token adapter with DLF-style disentanglement.

    Main outputs:
        pseudo_tokens: [B, N, text_dim], compatible with the frozen LLM.
        alpha: [B, 3], sample-wise modality reliability weights.
    """

    def __init__(self, text_dim=4096, hidden_dim=256, pseudo_tokens=4, num_heads=4,
                 tau=0.5, dropout=0.1, disentangle_margin=0.2,
                 loss_rec_weight=1.0, loss_spec_weight=1.0,
                 loss_metric_weight=0.1, loss_orth_weight=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).')

        self.hidden_dim = hidden_dim
        self.pseudo_tokens = pseudo_tokens
        self.tau = tau

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.disentangler = DLFDisentangler(
            hidden_dim=hidden_dim,
            dropout=dropout,
            margin=disentangle_margin,
            loss_rec_weight=loss_rec_weight,
            loss_spec_weight=loss_spec_weight,
            loss_metric_weight=loss_metric_weight,
            loss_orth_weight=loss_orth_weight,
        )

        # z_t/z_a/z_v -> alpha_t/alpha_a/alpha_v.
        # z_m is [shared semantic anchor, modality-private cues] compressed back to hidden_dim.
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        # unimodal auxiliary heads for label-aware router supervision
        self.head_t = nn.Linear(hidden_dim, 1)
        self.head_a = nn.Linear(hidden_dim, 1)
        self.head_v = nn.Linear(hidden_dim, 1)

        # modality-specific pseudo-token generators over disentangled representations
        self.token_gen_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pseudo_tokens * hidden_dim),
        )
        self.token_gen_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pseudo_tokens * hidden_dim),
        )
        self.token_gen_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pseudo_tokens * hidden_dim),
        )

        # primary-guided token-level enhancement in the pseudo-token space
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # project enhanced pseudo tokens back to LLM embedding dimension
        self.to_llm_dim = nn.Sequential(
            nn.Linear(hidden_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

    def _text_pool(self, text_embed):
        # mean pooling; text_embed is the ChatGLM embedding sequence [B, L, text_dim]
        return text_embed.mean(dim=1)

    def _make_tokens(self, generator, h):
        B = h.size(0)
        p = generator(h)
        p = p.view(B, self.pseudo_tokens, self.hidden_dim)
        return p

    def forward(self, text_embed, audio_h, video_h, labels=None):
        B = audio_h.size(0)

        h_t = self.text_proj(self._text_pool(text_embed))  # [B, hidden_dim]
        h_a = audio_h                                     # [B, hidden_dim]
        h_v = video_h                                     # [B, hidden_dim]

        dis_out = self.disentangler(h_t, h_a, h_v)
        z_t, z_a, z_v = dis_out['z_t'], dis_out['z_a'], dis_out['z_v']

        router_input = torch.cat([z_t, z_a, z_v], dim=-1)
        alpha = torch.softmax(self.router(router_input), dim=-1)  # [B, 3]

        P_t = self._make_tokens(self.token_gen_t, z_t)
        P_a = self._make_tokens(self.token_gen_a, z_a)
        P_v = self._make_tokens(self.token_gen_v, z_v)

        alpha_t = alpha[:, 0].view(B, 1, 1)
        alpha_a = alpha[:, 1].view(B, 1, 1)
        alpha_v = alpha[:, 2].view(B, 1, 1)

        # Dynamic token routing in the LLM pseudo-token space.
        # This soft primary representation is differentiable and more stable than argmax.
        P_dyn = alpha_t * P_t + alpha_a * P_a + alpha_v * P_v

        # Primary-guided token-level cross-modal enhancement.
        # Query: dynamically routed primary representation; Key/Value: all modality tokens.
        P_all = torch.cat([P_t, P_a, P_v], dim=1)
        P_attn, attn = self.cross_attn(query=P_dyn, key=P_all, value=P_all)
        P_enh = self.attn_norm(P_dyn + P_attn)
        P_enh = self.ffn_norm(P_enh + self.ffn(P_enh))

        pseudo_tokens = self.to_llm_dim(P_enh)  # [B, pseudo_tokens, text_dim]

        out = {
            'pseudo_tokens': pseudo_tokens,
            'alpha': alpha,
            'attn': attn,
            'P_t': P_t,
            'P_a': P_a,
            'P_v': P_v,
            'P_dyn': P_dyn,
            'disentangle_loss': dis_out['disentangle_loss'],
            'rec_loss': dis_out['rec_loss'],
            'spec_loss': dis_out['spec_loss'],
            'metric_loss': dis_out['metric_loss'],
            'orth_loss': dis_out['orth_loss'],
            'sh_fused': dis_out['sh_fused'],
        }

        if labels is not None:
            labels = labels.view(-1, 1).float()
            pred_t = self.head_t(z_t)
            pred_a = self.head_a(z_a)
            pred_v = self.head_v(z_v)

            uni_loss = (
                F.l1_loss(pred_t.float(), labels) +
                F.l1_loss(pred_a.float(), labels) +
                F.l1_loss(pred_v.float(), labels)
            )

            # Build a soft reliability target q from unimodal prediction errors.
            # Stop-gradient prevents q from becoming a moving target for the heads.
            with torch.no_grad():
                e_t = torch.abs(pred_t.float() - labels)
                e_a = torch.abs(pred_a.float() - labels)
                e_v = torch.abs(pred_v.float() - labels)
                e = torch.cat([e_t, e_a, e_v], dim=-1)
                q = torch.softmax(-e / self.tau, dim=-1)

            router_loss = F.kl_div(
                torch.log(alpha.float().clamp_min(1e-8)),
                q.float(),
                reduction='batchmean',
            )

            out['uni_loss'] = uni_loss
            out['router_loss'] = router_loss
            out['q'] = q
            out['pred_t'] = pred_t
            out['pred_a'] = pred_a
            out['pred_v'] = pred_v

        return out
