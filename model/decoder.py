import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class KGAnchoredPathDecoder(nn.Module):
    def __init__(self, input_dim, num_paths=64, path_dim=256, class_num=65, dropout_clf=0.4):
        super(KGAnchoredPathDecoder, self).__init__()
        self.num_paths = num_paths
        self.path_dim = path_dim
        
        self.path_memory = nn.Parameter(torch.randn(num_paths, path_dim))
        nn.init.orthogonal_(self.path_memory)
        
        self.left_proj = nn.Linear(input_dim, path_dim)
        self.right_proj = nn.Linear(input_dim, path_dim)
        self.agg_proj = nn.Linear(path_dim, input_dim)

        self.kg_rel_proj = nn.Linear(input_dim, path_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim * 2), # Input: [Left, Right, Path_Context]
            nn.BatchNorm1d(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_clf),    # S1: 0.2, S2: 0.3, S3: 0.4
            nn.Linear(input_dim * 2, class_num)
        )

        self._last_alpha = None

    def init_from_kg_relations(self, kg_edge_emb):
        with torch.no_grad():
            emb_proj = self.kg_rel_proj(kg_edge_emb)
            emb_np = emb_proj.cpu().detach().numpy()
            km = KMeans(n_clusters=self.num_paths, n_init=10, random_state=42)
            km.fit(emb_np)
            centers = torch.tensor(
                km.cluster_centers_, dtype=torch.float, device=kg_edge_emb.device
            )
            self.path_memory.data.copy_(centers)
        print(f"[Decoder] path_memory initialized from {kg_edge_emb.shape[0]} KG relations"
              f"-> {self.num_paths} prototypes")

    def forward(self, left_emb, right_emb):
        l_curr = torch.tanh(self.left_proj(left_emb))
        r_curr = torch.tanh(self.right_proj(right_emb))
        score = torch.matmul(l_curr, self.path_memory.t()) + torch.matmul(r_curr, self.path_memory.t())
        alpha = F.softmax(score, dim=-1)

        self._last_alpha = alpha.detach()
        
        path_context = self.agg_proj(torch.matmul(alpha, self.path_memory))
        combined = torch.cat([left_emb, right_emb, path_context], dim=-1)
        pred = self.classifier(combined)
        
        return pred, alpha
    
    def _forward_with_path_mask(self, left_emb, right_emb, mask):
        l_curr = torch.tanh(self.left_proj(left_emb))
        r_curr = torch.tanh(self.right_proj(right_emb))
        logits = torch.matmul(l_curr, self.path_memory.t()) + torch.matmul(r_curr, self.path_memory.t())
        logits = logits.masked_fill(mask, -1e9)
        alpha_cf = F.softmax(logits, dim=-1)

        path_context = self.agg_proj(torch.matmul(alpha_cf, self.path_memory))
        combined = torch.cat([left_emb, right_emb, path_context], dim=-1)
        pred_cf = self.classifier(combined)
        return pred_cf

    def get_orthogonality_loss(self):
        norm_p = F.normalize(self.path_memory, p=2, dim=1)
        cos_mat = torch.matmul(norm_p, norm_p.t())
        eye = torch.eye(self.num_paths, device=self.path_memory.device)
        ortho_loss = torch.norm(cos_mat - eye, p='fro')
        return ortho_loss
    
    def get_kg_alignment_loss(self, kg_edge_emb):
        kg_proj = self.kg_rel_proj(kg_edge_emb)
        norm_p = F.normalize(self.path_memory, p=2, dim=1)
        norm_r = F.normalize(kg_proj, p=2, dim=1)
        sim = torch.matmul(norm_p, norm_r.t())
        align_loss = (1.0 - sim.max(dim=1).values).mean()
        return align_loss

    def get_hierarchical_causal_loss(self, 
                                     left_emb, 
                                     right_emb, 
                                     pred_orig, 
                                     sub_cf_pred,
                                     top_ratio=0.1, 
                                     tau=0.4):
        assert self._last_alpha is not None, "Last alpha values are not available"

        prob_orig = F.softmax(pred_orig, dim=-1)
        prob_cf_s = F.softmax(sub_cf_pred.float(), dim=-1).detach()
        delta_s = torch.norm(prob_orig - prob_cf_s, p=1, dim=-1).mean()

        alpha = self._last_alpha
        m = max(1, int(self.num_paths * top_ratio))
        _, top_idx = torch.topk(alpha, m, dim=-1)
        mask = torch.zeros_like(alpha, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

        with torch.no_grad():
            pred_cf_p = self._forward_with_path_mask(left_emb.detach(), right_emb.detach(), mask)

        prob_cf_p = F.softmax(pred_cf_p.float(), dim=-1).detach()
        delta_p = torch.norm(prob_orig - prob_cf_p, p=1, dim=-1).mean()

        with torch.no_grad():
            total = delta_s.detach() + delta_p.detach() + 1e-6
            lambda_s = delta_s.detach() / total
            lambda_p = delta_p.detach() / total

        loss_s = torch.clamp(tau - delta_s, min=0.0)
        loss_p = torch.clamp(tau - delta_p, min=0.0)
        causal_loss = lambda_s * loss_s + lambda_p * loss_p

        return causal_loss

