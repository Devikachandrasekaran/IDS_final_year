# backend/models/flow_model.py
import torch
import torch.nn as nn
import math

# Simple TGN-like components (copy of your Colab architecture)
class SimpleMemory(nn.Module):
    def __init__(self, input_dim, mem_dim):
        super().__init__()
        self.grucell = nn.GRUCell(input_dim, mem_dim)
    def forward(self, mem, msg):
        return self.grucell(msg, mem)

class GraphAttentionContext(nn.Module):
    def __init__(self, msg_dim, att_dim):
        super().__init__()
        self.key = nn.Linear(msg_dim, att_dim)
        self.query = nn.Linear(msg_dim, att_dim)
        self.value = nn.Linear(msg_dim, msg_dim)
        self.scale = math.sqrt(att_dim)
    def forward(self, query_msg, neighbor_msgs):
        # neighbor_msgs: (B, K, msg_dim)
        if neighbor_msgs is None or neighbor_msgs.shape[1] == 0:
            return torch.zeros(query_msg.size(0), query_msg.size(1), device=query_msg.device)
        Q = self.query(query_msg).unsqueeze(1)   # (B,1,att)
        K = self.key(neighbor_msgs)              # (B,K,att)
        V = self.value(neighbor_msgs)            # (B,K,msg_dim)
        att = torch.softmax((Q * K).sum(-1) / self.scale, dim=1).unsqueeze(-1)  # (B,K,1)
        ctx = (att * V).sum(1)  # (B,msg_dim)
        return ctx

class FlowGATTGNGRU(nn.Module):
    def __init__(self, feat_dim, mem_dim=32, att_dim=32, gru_hid=64):
        super().__init__()
        self.msg_proj = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.memory = SimpleMemory(64, mem_dim)
        self.gat_context = GraphAttentionContext(64, att_dim)
        self.gru_in_dim = feat_dim + 64 + mem_dim
        self.gru = nn.GRU(self.gru_in_dim, gru_hid, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hid, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, window_feats, neighbor_msgs=None, mem=None):
        # window_feats: (B, L, feat_dim)
        B, L, F = window_feats.shape
        all_msgs = self.msg_proj(window_feats.view(B*L, F)).view(B, L, -1)
        if mem is None:
            mem = torch.zeros(B, 32, device=window_feats.device)
        last_msg = all_msgs[:, -1, :]
        new_mem = self.memory(mem, last_msg)
        ctx = self.gat_context(last_msg, neighbor_msgs) if neighbor_msgs is not None else torch.zeros(B, 64, device=window_feats.device)
        mem_expand = new_mem.unsqueeze(1).repeat(1, L, 1)
        gru_input = torch.cat([window_feats, all_msgs, mem_expand], dim=-1)
        out, _ = self.gru(gru_input)
        last_h = out[:, -1, :]
        prob = self.classifier(last_h).squeeze(-1)   # (B,)
        return prob, new_mem
