# # backend/models/load_model.py
# import torch
# import joblib
# import os
# from .flow_model import FlowGATTGNGRU  # Ensure flow_model.py exists if needed

# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
# MODEL_PATH = os.path.join(BASE_DIR, "..", "trained_models", "flowgattgn_gru.pth")
# SCALER_PATH = os.path.join(BASE_DIR, "..", "trained_models", "scaler.save")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feat_dim = 8

# # Load model
# model = FlowGATTGNGRU(feat_dim).to(device)
# state = torch.load(MODEL_PATH, map_location=device)
# if isinstance(state, dict):
#     model.load_state_dict(state)
# else:
#     model = state.to(device)
# model.eval()

# # Load scaler
# scaler = joblib.load(SCALER_PATH)


# backend/models/load_model.py
# import torch
# import torch.nn as nn
# import math
# import joblib
# import os

# # ----------------- Model Classes -----------------
# class SimpleMemory(nn.Module):
#     def __init__(self, input_dim, mem_dim):
#         super().__init__()
#         self.grucell = nn.GRUCell(input_dim, mem_dim)

#     def forward(self, mem, msg):
#         return self.grucell(msg, mem)

# class GraphAttentionContext(nn.Module):
#     def __init__(self, msg_dim, att_dim):
#         super().__init__()
#         self.key = nn.Linear(msg_dim, att_dim)
#         self.query = nn.Linear(msg_dim, att_dim)
#         self.value = nn.Linear(msg_dim, msg_dim)
#         self.scale = math.sqrt(att_dim)

#     def forward(self, query_msg, neighbor_msgs):
#         if neighbor_msgs is None or neighbor_msgs.shape[1] == 0:
#             return torch.zeros(query_msg.size(0), query_msg.size(1), device=query_msg.device)
#         Q = self.query(query_msg).unsqueeze(1)   # (B,1,att)
#         K = self.key(neighbor_msgs)              # (B,K,att)
#         V = self.value(neighbor_msgs)            # (B,K,msg_dim)
#         att = torch.softmax((Q * K).sum(-1) / self.scale, dim=1).unsqueeze(-1)  # (B,K,1)
#         ctx = (att * V).sum(1)  # (B,msg_dim)
#         return ctx

# class FlowGATTGNGRU(nn.Module):
#     def __init__(self, feat_dim, mem_dim=32, att_dim=32, gru_hid=64):
#         super().__init__()
#         self.msg_proj = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, 64))
#         self.memory = SimpleMemory(64, mem_dim)
#         self.gat_context = GraphAttentionContext(64, att_dim)
#         self.gru_in_dim = feat_dim + 64 + mem_dim
#         self.gru = nn.GRU(self.gru_in_dim, gru_hid, batch_first=True)
#         self.classifier = nn.Sequential(
#             nn.Linear(gru_hid, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, window_feats, neighbor_msgs=None, mem=None):
#         B, L, F = window_feats.shape
#         all_msgs = self.msg_proj(window_feats.view(B*L, F)).view(B, L, -1)
#         if mem is None:
#             mem = torch.zeros(B, 32, device=window_feats.device)
#         last_msg = all_msgs[:, -1, :]
#         new_mem = self.memory(mem, last_msg)
#         ctx = self.gat_context(last_msg, neighbor_msgs) if neighbor_msgs is not None else torch.zeros(B, 64, device=window_feats.device)
#         mem_expand = new_mem.unsqueeze(1).repeat(1, L, 1)
#         gru_input = torch.cat([window_feats, all_msgs, mem_expand], dim=-1)
#         out, _ = self.gru(gru_input)
#         last_h = out[:, -1, :]
#         prob = self.classifier(last_h).squeeze(-1)   # (B,)
#         return prob, new_mem

# # ----------------- Paths -----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/models
# MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "trained_models", "flowgattgn_gru.pth")
# SCALER_PATH = os.path.join(BASE_DIR, "..", "..", "trained_models", "scaler.save")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feat_dim = 8

# # ----------------- Load Model & Scaler -----------------
# model = FlowGATTGNGRU(feat_dim).to(device)
# state = torch.load(MODEL_PATH, map_location=device)
# if isinstance(state, dict):
#     model.load_state_dict(state)
# else:
#     model = state.to(device)
# model.eval()

# scaler = joblib.load(SCALER_PATH)






# # backend/models/load_model.py
# import os
# import torch
# import joblib

# # Determine backend directory
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # <project>/backend/models
# BACKEND_DIR = os.path.dirname(THIS_DIR)                      # <project>/backend
# TRAINED_DIR = os.path.join(BACKEND_DIR, "trained_models")

# # Model & scaler filenames (update names if different)
# MODEL_FILENAME = "flowgattgn_gru.pth"
# SCALER_FILENAME = "scaler.save"

# MODEL_PATH = os.path.join(TRAINED_DIR, MODEL_FILENAME)
# SCALER_PATH = os.path.join(TRAINED_DIR, SCALER_FILENAME)

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Minimal model class definition used at load-time
# # This must match the class used when the model was saved.
# # If your project has the model class elsewhere, import it instead.
# import torch.nn as nn
# import math

# class SimpleMemory(nn.Module):
#     def __init__(self, input_dim, mem_dim):
#         super().__init__()
#         self.grucell = nn.GRUCell(input_dim, mem_dim)
#     def forward(self, mem, msg):
#         return self.grucell(msg, mem)

# class GraphAttentionContext(nn.Module):
#     def __init__(self, msg_dim, att_dim):
#         super().__init__()
#         self.key = nn.Linear(msg_dim, att_dim)
#         self.query = nn.Linear(msg_dim, att_dim)
#         self.value = nn.Linear(msg_dim, msg_dim)
#         self.scale = math.sqrt(att_dim)
#     def forward(self, query_msg, neighbor_msgs):
#         if neighbor_msgs is None or neighbor_msgs.shape[1] == 0:
#             return torch.zeros(query_msg.size(0), query_msg.size(1), device=query_msg.device)
#         Q = self.query(query_msg).unsqueeze(1)
#         K = self.key(neighbor_msgs)
#         V = self.value(neighbor_msgs)
#         att = torch.softmax((Q*K).sum(-1)/self.scale, dim=1).unsqueeze(-1)
#         return (att*V).sum(1)

# class FlowGATTGNGRU(nn.Module):
#     def __init__(self, feat_dim, mem_dim=32, att_dim=32, gru_hid=64):
#         super().__init__()
#         self.msg_proj = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64,64))
#         self.memory = SimpleMemory(64, mem_dim)
#         self.gat_context = GraphAttentionContext(64, att_dim)
#         self.gru_in_dim = feat_dim + 64 + mem_dim
#         self.gru = nn.GRU(self.gru_in_dim, gru_hid, batch_first=True)
#         self.classifier = nn.Sequential(nn.Linear(gru_hid,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
#     def forward(self, window_feats, neighbor_msgs=None, mem=None):
#         B,L,F = window_feats.shape
#         all_msgs = self.msg_proj(window_feats.view(B*L,F)).view(B,L,-1)
#         if mem is None:
#             mem = torch.zeros(B,32,device=window_feats.device)
#         last_msg = all_msgs[:,-1,:]
#         new_mem = self.memory(mem,last_msg)
#         ctx = self.gat_context(last_msg, neighbor_msgs) if neighbor_msgs is not None else torch.zeros(B,64,device=window_feats.device)
#         mem_expand = new_mem.unsqueeze(1).repeat(1,L,1)
#         gru_input = torch.cat([window_feats, all_msgs, mem_expand], dim=-1)
#         out,_ = self.gru(gru_input)
#         last_h = out[:,-1,:]
#         prob = self.classifier(last_h).squeeze(-1)
#         return prob,new_mem

# # Load model & scaler safely
# model = None
# scaler = None

# def _safe_load():
#     global model, scaler
#     # Make sure files exist
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
#     if not os.path.exists(SCALER_PATH):
#         raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

#     # Try to load PyTorch model (it may be either state_dict or full model)
#     try:
#         state = torch.load(MODEL_PATH, map_location=device)
#         if isinstance(state, dict):
#             m = FlowGATTGNGRU(feat_dim=8)  # ensure feat_dim matches training
#             m.load_state_dict(state)
#             model = m.to(device)
#         else:
#             model = state.to(device)
#         model.eval()
#     except Exception as e:
#         raise RuntimeError(f"Failed to load model: {e}")

#     # Load scaler
#     try:
#         scaler_local = joblib.load(SCALER_PATH)
#         scaler = scaler_local
#     except Exception as e:
#         raise RuntimeError(f"Failed to load scaler: {e}")

# # Load at import time (so other modules can import model, scaler, device)
# try:
#     _safe_load()
# except Exception as e:
#     # Re-raise so error is visible
#     raise






import os
import torch
import joblib
import torch.nn as nn
import math

# ============================================================
# 1. Determine correct project and trained_model paths
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # backend/models
BACKEND_DIR = os.path.dirname(THIS_DIR)                          # backend
PROJECT_DIR = os.path.dirname(BACKEND_DIR)                       # ids_react_project

# trained_models is OUTSIDE backend, so go up one level
TRAINED_DIR = os.path.join(PROJECT_DIR, "trained_models")

MODEL_FILENAME = "flowgattgn_gru.pth"
SCALER_FILENAME = "scaler.save"

MODEL_PATH = os.path.join(TRAINED_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(TRAINED_DIR, SCALER_FILENAME)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Model class definitions (must match training-time classes)
# ============================================================

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
        if neighbor_msgs is None or neighbor_msgs.shape[1] == 0:
            return torch.zeros(query_msg.size(0), query_msg.size(1), device=query_msg.device)

        Q = self.query(query_msg).unsqueeze(1)
        K = self.key(neighbor_msgs)
        V = self.value(neighbor_msgs)

        att = torch.softmax((Q * K).sum(-1) / self.scale, dim=1).unsqueeze(-1)
        return (att * V).sum(1)


class FlowGATTGNGRU(nn.Module):
    def __init__(self, feat_dim, mem_dim=32, att_dim=32, gru_hid=64):
        super().__init__()

        self.msg_proj = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

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
        B, L, F = window_feats.shape

        all_msgs = self.msg_proj(window_feats.view(B * L, F)).view(B, L, -1)

        if mem is None:
            mem = torch.zeros(B, 32, device=window_feats.device)

        last_msg = all_msgs[:, -1, :]
        new_mem = self.memory(mem, last_msg)

        ctx = (
            self.gat_context(last_msg, neighbor_msgs)
            if neighbor_msgs is not None
            else torch.zeros(B, 64, device=window_feats.device)
        )

        mem_expand = new_mem.unsqueeze(1).repeat(1, L, 1)

        gru_input = torch.cat([window_feats, all_msgs, mem_expand], dim=-1)
        out, _ = self.gru(gru_input)

        last_h = out[:, -1, :]
        prob = self.classifier(last_h).squeeze(-1)

        return prob, new_mem


# ============================================================
# 3. Safe loading
# ============================================================

model = None
scaler = None

def _safe_load():
    global model, scaler

    # Ensure files exist
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

    # Load model
    try:
        state = torch.load(MODEL_PATH, map_location=device)

        if isinstance(state, dict):
            m = FlowGATTGNGRU(feat_dim=8)  # <-- adjust if your model uses different feature size
            m.load_state_dict(state)
            model = m.to(device)
        else:
            model = state.to(device)

        model.eval()

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Load scaler
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")


# Load automatically when imported
try:
    _safe_load()
except Exception as e:
    raise
