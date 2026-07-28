import torch
import torch.nn.functional as F
import torch.nn as nn
from model.hgnn import HGNN
from model.hypergraph import SubHypergraphEncoder
from model.decoder import KGAnchoredPathDecoder


class SubstructurePairCoAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2):
        super(SubstructurePairCoAttention, self).__init__()
        self.attn_AB = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_BA = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_A = nn.LayerNorm(hidden_dim)
        self.norm_B = nn.LayerNorm(hidden_dim)
        self.ffn_A = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_B = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_A2 = nn.LayerNorm(hidden_dim)
        self.norm_B2 = nn.LayerNorm(hidden_dim)

        self.gate_mlp_A = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.gate_mlp_B = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb_A, emb_B, sub_A, mask_A, sub_B, mask_B):
        qA = emb_A.unsqueeze(1)
        qB = emb_B.unsqueeze(1)

        A_cross, attn_A2B = self.attn_AB(qA, sub_B, sub_B, key_padding_mask=~mask_B)
        A_refined = self.norm_A((qA + A_cross).squeeze(1))
        A_refined = self.norm_A2(A_refined + self.ffn_A(A_refined))

        B_cross, attn_B2A = self.attn_BA(qB, sub_A, sub_A, key_padding_mask=~mask_A)
        B_refined = self.norm_B((qB + B_cross).squeeze(1))
        B_refined = self.norm_B2(B_refined + self.ffn_B(B_refined))

        pair_AB = torch.cat([emb_A, A_refined], dim=-1)
        pair_BA = torch.cat([emb_B, B_refined], dim=-1)

        g_A = torch.sigmoid(self.gate_mlp_A(pair_AB))
        g_B = torch.sigmoid(self.gate_mlp_B(pair_BA))

        A_out = emb_A + g_A * (A_refined - emb_A)
        B_out = emb_B + g_B * (B_refined - emb_B)

        return A_out, B_out, attn_A2B.squeeze(1), attn_B2A.squeeze(1)


class FaithDDI(nn.Module):
    def __init__(self,
                 kg_g,
                 smiles,
                 hidden_dim,
                 num_layer,
                 class_num,
                 condition,
                 dropout_hyper=0.4,
                 dropout_clf=0.4,
                 dropout_coattn=0.2,
                 dropout_kg_emb=0.3,
                 ):
        super(FaithDDI, self).__init__()

        self.smiles = smiles
        self.device = kg_g.device
        self.drug_num = len(smiles)

        self.kg = HGNN(kg_g, kg_g.edata['edges'], kg_g.ndata['nodes'], hidden_dim, num_layer)
        self.kg_fc = nn.Linear(hidden_dim, hidden_dim)

        self.hyper_mol = SubHypergraphEncoder(smiles, hidden_dim, num_layer, self.device, dropout=dropout_hyper)
        self.mol_fc = nn.Linear(hidden_dim, hidden_dim)

        self.kg_emb_dropout = nn.Dropout(p=dropout_kg_emb)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        if condition == 'S1':
            gate_value = -3.0
        elif condition == 'S2':
            gate_value = -1.0
        else:
            gate_value = -0.5

        self.co_attn = SubstructurePairCoAttention(hidden_dim, num_heads=4, dropout=dropout_coattn, gate=gate_value)    # S1: 0.1, S2: 0.15, S3: 0.2

        self.decoder = KGAnchoredPathDecoder(hidden_dim, num_paths=64, path_dim=256, class_num=class_num, dropout_clf=dropout_clf)

        self._kg_cache = None

        kg_edge_emb = self.kg.gat_layers[0].edge_embedding.weight.detach()
        self.decoder.init_from_kg_relations(kg_edge_emb)

    def cache_kg_embedding(self):
        kg_raw = self.kg_fc(self.kg())
        if self.training:
            kg_raw = self.kg_emb_dropout(kg_raw)
        self._kg_cache = kg_raw

    def get_drug_embedding(self, ids, mask_substructures=None):
        mol_all = self.mol_fc(self.hyper_mol(mask_indices=mask_substructures))
        
        assert self._kg_cache is not None, "KG embeddings not cached. Call cache_kg_embedding() before training."

        kg_all = self._kg_cache

        if ids is None:
            mol_emb = mol_all
            kg_emb = kg_all[:self.drug_num]
        else:
            mol_emb = mol_all[ids]
            kg_emb = kg_all[ids]

        combined = torch.cat([kg_emb, mol_emb], dim=-1)
        gate = self.fusion_gate(combined)
        fused_emb = self.fusion_proj(combined) * gate

        return fused_emb

    def forward(self, left, right, mask_substructures=None, return_alpha=True):
        left_emb = self.get_drug_embedding(left, mask_substructures)
        right_emb = self.get_drug_embedding(right, mask_substructures)

        left_sub, left_mask = self.hyper_mol.get_substructures_embeddings(left)
        right_sub, right_mask = self.hyper_mol.get_substructures_embeddings(right)

        left_emb, right_emb, attn_l2r, attn_r2l = self.co_attn(
            left_emb, right_emb, 
            left_sub, left_mask, 
            right_sub, right_mask)

        pred, alpha = self.decoder(left_emb, right_emb)

        if return_alpha:
            self.last_attn_l2r = attn_l2r.detach()
            self.last_attn_r2l = attn_r2l.detach()
            return pred, alpha, left_emb, right_emb
        return pred, None, None, None
