# backend/utils/consumer_predict.py
# import json
# import torch
# import joblib
# from confluent_kafka import Consumer
# import os

# # -------- Paths and device --------
# MODEL_PATH = os.path.join("..", "..", "trained_models", "flowgattgn_gru.pth")
# SCALER_PATH = os.path.join("..", "..", "trained_models", "scaler.save")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feat_dim = 8

# # -------- Model class definitions --------
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
#         if neighbor_msgs is None or neighbor_msgs.shape[1]==0:
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

# # -------- Load model and scaler --------
# model = FlowGATTGNGRU(feat_dim).to(device)
# state = torch.load(MODEL_PATH, map_location=device)
# if isinstance(state, dict):
#     model.load_state_dict(state)
# else:
#     model = state.to(device)
# model.eval()
# scaler = joblib.load(SCALER_PATH)

# # -------- Kafka consumer config --------
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "test"
# GROUP_ID = "ids-consumer-group"

# consumer = Consumer({
#     'bootstrap.servers': KAFKA_BOOTSTRAP,
#     'group.id': GROUP_ID,
#     'auto.offset.reset': 'earliest'
# })
# consumer.subscribe([TOPIC])

# # -------- Preprocess flow --------
# def preprocess_flow(flow):
#     features = [
#         float(flow.get('src_ip_h',0)),
#         float(flow.get('dst_ip_h',0)),
#         float(flow.get('src_port',0)),
#         float(flow.get('dst_port',0)),
#         float(flow.get('protocol_num',0)),
#         float(flow.get('pkt_count',0)),
#         float(flow.get('byte_count',0)),
#         float(flow.get('duration',0))
#     ]
#     x = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(device)
#     x_scaled = torch.tensor(scaler.transform(x.squeeze(1).cpu()), dtype=torch.float32).unsqueeze(1).to(device)
#     return x_scaled

# # -------- Main loop --------
# print("Kafka consumer started. Listening for live flows...")

# try:
#     while True:
#         msg = consumer.poll(1.0)
#         if msg is None:
#             continue
#         if msg.error():
#             print("Kafka error:", msg.error())
#             continue
#         try:
#             flow_json = json.loads(msg.value().decode("utf-8"))
#         except:
#             continue
#         X_input = preprocess_flow(flow_json)
#         with torch.no_grad():
#             prob,_ = model(X_input)
#             pred_label = int(prob.item() > 0.5)
#         print(f"Flow {flow_json.get('src_ip')} -> {flow_json.get('dst_ip')} | Predicted: {pred_label}")

# except KeyboardInterrupt:
#     print("Stopping consumer...")
# finally:
#     consumer.close()

# import sys
# import json
# import torch
# import joblib
# from confluent_kafka import Consumer
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from stats_store import update_stats  # import the stats updater

# # -------- Paths and device --------
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
# MODEL_PATH = os.path.join(BASE_DIR, "..", "trained_models", "flowgattgn_gru.pth")
# SCALER_PATH = os.path.join(BASE_DIR, "..", "trained_models", "scaler.save")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feat_dim = 8

# # -------- Model class definitions --------
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
#         if neighbor_msgs is None or neighbor_msgs.shape[1]==0:
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

# # -------- Load model and scaler --------
# model = FlowGATTGNGRU(feat_dim).to(device)
# state = torch.load(MODEL_PATH, map_location=device)
# if isinstance(state, dict):
#     model.load_state_dict(state)
# else:
#     model = state.to(device)
# model.eval()
# scaler = joblib.load(SCALER_PATH)

# # -------- Kafka consumer config --------
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "test"
# GROUP_ID = "ids-consumer-group"

# consumer = Consumer({
#     'bootstrap.servers': KAFKA_BOOTSTRAP,
#     'group.id': GROUP_ID,
#     'auto.offset.reset': 'earliest'
# })
# consumer.subscribe([TOPIC])

# # -------- Preprocess flow --------
# def preprocess_flow(flow):
#     features = [
#         float(flow.get('src_ip_h',0)),
#         float(flow.get('dst_ip_h',0)),
#         float(flow.get('src_port',0)),
#         float(flow.get('dst_port',0)),
#         float(flow.get('protocol_num',0)),
#         float(flow.get('pkt_count',0)),
#         float(flow.get('byte_count',0)),
#         float(flow.get('duration',0))
#     ]
#     x = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(device)
#     x_scaled = torch.tensor(scaler.transform(x.squeeze(1).cpu()), dtype=torch.float32).unsqueeze(1).to(device)
#     return x_scaled

# # -------- Main loop --------
# print("Kafka consumer started. Listening for live flows...")

# try:
#     while True:
#         msg = consumer.poll(1.0)
#         if msg is None:
#             continue
#         if msg.error():
#             print("Kafka error:", msg.error())
#             continue
#         try:
#             flow_json = json.loads(msg.value().decode("utf-8"))
#         except:
#             continue

#         X_input = preprocess_flow(flow_json)
#         with torch.no_grad():
#             prob,_ = model(X_input)
#             pred_label = int(prob.item() > 0.5)

#         # ---- Update shared stats for dashboard ----
#         update_stats(pred_label)

#         # ---- Print live flow prediction ----
#         pred_label = int(prob.item() > 0.5)
#         update_stats(pred_label) 
#         print(f"Flow {flow_json.get('src_ip')} -> {flow_json.get('dst_ip')} | Predicted: {pred_label}")

# except KeyboardInterrupt:
#     print("Stopping consumer...")

# finally:
#     consumer.close()




# import sys
# import os
# import json
# import torch
# import joblib
# from confluent_kafka import Consumer
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from ids_react_project.backend.utils.stats_store import update_stats  # thread-safe stats update

# # ---------------- Paths ----------------
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
# MODEL_PATH = os.path.join(BASE_DIR, "..", "trained_models", "flowgattgn_gru.pth")
# SCALER_PATH = os.path.join(BASE_DIR, "..", "trained_models", "scaler.save")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# feat_dim = 8

# # ---------------- Model ----------------
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
#         if neighbor_msgs is None or neighbor_msgs.shape[1]==0:
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

# # ---------------- Load model & scaler ----------------
# model = FlowGATTGNGRU(feat_dim).to(device)
# state = torch.load(MODEL_PATH, map_location=device)
# if isinstance(state, dict):
#     model.load_state_dict(state)
# else:
#     model = state.to(device)
# model.eval()
# scaler = joblib.load(SCALER_PATH)

# # ---------------- Kafka ----------------
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "test"
# GROUP_ID = "ids-consumer-group"

# consumer = Consumer({
#     'bootstrap.servers': KAFKA_BOOTSTRAP,
#     'group.id': GROUP_ID,
#     'auto.offset.reset': 'earliest'
# })
# consumer.subscribe([TOPIC])

# # ---------------- Preprocess ----------------
# def preprocess_flow(flow):
#     features = [
#         float(flow.get('src_ip_h',0)),
#         float(flow.get('dst_ip_h',0)),
#         float(flow.get('src_port',0)),
#         float(flow.get('dst_port',0)),
#         float(flow.get('protocol_num',0)),
#         float(flow.get('pkt_count',0)),
#         float(flow.get('byte_count',0)),
#         float(flow.get('duration',0))
#     ]
#     x = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(device)
#     x_scaled = torch.tensor(scaler.transform(x.squeeze(1).cpu()), dtype=torch.float32).unsqueeze(1).to(device)
#     return x_scaled

# # ---------------- Main loop ----------------
# print("✅ Kafka consumer started. Listening for live flows...")

# try:
#     while True:
#         msg = consumer.poll(1.0)
#         if msg is None:
#             continue
#         if msg.error():
#             print("Kafka error:", msg.error())
#             continue
#         try:
#             flow_json = json.loads(msg.value().decode("utf-8"))
#         except:
#             continue

#         X_input = preprocess_flow(flow_json)
#         with torch.no_grad():
#             prob,_ = model(X_input)
#             pred_label = int(prob.item() > 0.5)

#         # Update stats for dashboard
#         update_stats(flow_json, pred_label)

#         # Print live flow
#         print(f"Flow {flow_json.get('src_ip')} -> {flow_json.get('dst_ip')} | Predicted: {pred_label}")

# except KeyboardInterrupt:
#     print("Stopping consumer...")

# finally:
#     consumer.close()





# backend/utils/consumer_predict.py
"""
Run this from the backend folder:
    cd <project>/ids_react_project/backend
    python -m utils.consumer_predict

This script consumes JSON flows from Kafka topic 'test', preprocesses, predicts using the loaded model,
and updates stats via stats_store.update_stats() for the Flask dashboard.
"""

# import os
# import sys
# import json
# import time
# import traceback
# from confluent_kafka import Consumer, KafkaException
# import torch
# from models.load_model import model, scaler, device
# from utils.preprocess import flow_to_feature_vector, make_tensor_from_vec
# from utils.stats_store import update_stats

# # Kafka config
# KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
# TOPIC = os.environ.get("KAFKA_TOPIC", "test")
# GROUP_ID = os.environ.get("KAFKA_GROUP", "ids-consumer-group")

# consumer = Consumer({
#     'bootstrap.servers': KAFKA_BOOTSTRAP,
#     'group.id': GROUP_ID,
#     'auto.offset.reset': 'earliest'
# })
# consumer.subscribe([TOPIC])

# print("✅ Kafka consumer started. Listening for live flows on topic:", TOPIC)
# print(f"Model device: {device}, model loaded: {model is not None}")

# def safe_predict(flow_json):
#     """
#     Preprocess -> predict -> return (pred_label, prob)
#     """
#     try:
#         vec = flow_to_feature_vector(flow_json)
#         tensor = make_tensor_from_vec(vec, scaler, device)
#         with torch.no_grad():
#             prob_tensor, _ = model(tensor)
#             prob = float(prob_tensor.cpu().numpy().ravel()[0])
#             pred_label = int(prob > 0.5)
#         return pred_label, prob
#     except Exception as e:
#         # detailed logging for debugging
#         print("Prediction error:", e)
#         traceback.print_exc()
#         return 0, 0.0

# try:
#     while True:
#         try:
#             msg = consumer.poll(1.0)
#             if msg is None:
#                 continue
#             if msg.error():
#                 # print error but continue
#                 print("Kafka error:", msg.error())
#                 continue
#             raw = msg.value()
#             try:
#                 s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
#                 flow_json = json.loads(s)
#             except Exception as e:
#                 print("Received non-JSON or malformed message; skipping. error:", e)
#                 print("Raw payload:", raw)
#                 continue

#             # add timestamp to flow (optional)
#             if "ts" not in flow_json:
#                 flow_json["ts"] = time.time()

#             pred_label, prob = safe_predict(flow_json)

#             # update stats store (thread-safe)
#             try:
#                 update_stats(flow_json, pred_label)
#             except Exception as e:
#                 print("Failed to update stats:", e)

#             # logging
#             print(f"Flow {flow_json.get('src_ip')} -> {flow_json.get('dst_ip')} | Predicted: {pred_label} | prob: {prob:.4f}")

#         except KafkaException as ke:
#             print("Kafka exception, sleeping 1s:", ke)
#             time.sleep(1)
#         except KeyboardInterrupt:
#             print("KeyboardInterrupt: stopping consumer")
#             break
#         except Exception as e:
#             print("Unexpected error in consumer loop:", e)
#             traceback.print_exc()
#             time.sleep(1)

# except KeyboardInterrupt:
#     print("Stopping consumer...")

# finally:
#     consumer.close()
#     print("Consumer closed")




# import os
# import sys
# import json
# import time
# import traceback
# import torch
# from confluent_kafka import Consumer, KafkaException

# # ------------------------------------------------
# # FIX → Add project root to Python path
# # ------------------------------------------------
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
# sys.path.append(PROJECT_ROOT)

# # ------------------------------------------------
# # FIX → Correct full absolute imports
# # ------------------------------------------------
# from ids_react_project.backend.models.load_model import model, scaler, device
# from ids_react_project.backend.utils.preprocess import (
#     flow_to_feature_vector,
#     make_tensor_from_vec
# )
# from ids_react_project.backend.utils.stats_store import update_stats

# # ------------------------------------------------
# # Kafka Config
# # ------------------------------------------------
# KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
# TOPIC = os.environ.get("KAFKA_TOPIC", "test")
# GROUP_ID = os.environ.get("KAFKA_GROUP", "ids-consumer-group")

# consumer = Consumer({
#     'bootstrap.servers': KAFKA_BOOTSTRAP,
#     'group.id': GROUP_ID,
#     'auto.offset.reset': 'earliest'
# })

# consumer.subscribe([TOPIC])

# print("✅ Kafka Consumer Started")
# print("Topic:", TOPIC)
# print(f"Model Device: {device}, Model Loaded: {model is not None}")

# # ------------------------------------------------
# # Prediction Wrapper
# # ------------------------------------------------
# def safe_predict(flow_json):
#     try:
#         vec = flow_to_feature_vector(flow_json)
#         tensor = make_tensor_from_vec(vec, scaler, device)

#         with torch.no_grad():
#             prob_tensor, _ = model(tensor)
#             prob = float(prob_tensor.cpu().numpy().ravel()[0])
#             pred_label = int(prob > 0.5)

#         return pred_label, prob

#     except Exception as e:
#         print("🔥 Prediction Error:", e)
#         traceback.print_exc()
#         return 0, 0.0


# # ------------------------------------------------
# # Main Consumer Loop
# # ------------------------------------------------
# try:
#     while True:
#         try:
#             msg = consumer.poll(1.0)
#             if msg is None:
#                 continue

#             if msg.error():
#                 print("Kafka Error:", msg.error())
#                 continue

#             raw = msg.value()

#             try:
#                 s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
#                 flow_json = json.loads(s)
#             except Exception as e:
#                 print("⚠️ Malformed JSON:", e)
#                 print("Raw Message:", raw)
#                 continue

#             if "ts" not in flow_json:
#                 flow_json["ts"] = time.time()

#             pred_label, prob = safe_predict(flow_json)

#             try:
#                 update_stats(flow_json, pred_label)
#             except Exception as e:
#                 print("Stats Update Failed:", e)

#             print(
#                 f"Flow {flow_json.get('src_ip')} → {flow_json.get('dst_ip')} | "
#                 f"Prediction: {pred_label} | Prob: {prob:.4f}"
#             )

#         except KafkaException as ke:
#             print("Kafka Exception:", ke)
#             time.sleep(1)

#         except KeyboardInterrupt:
#             print("🛑 KeyboardInterrupt — Shutting Down")
#             break

#         except Exception as e:
#             print("🔥 Unexpected Consumer Error:", e)
#             traceback.print_exc()
#             time.sleep(1)

# except KeyboardInterrupt:
#     print("Stopping consumer...")

# finally:
#     consumer.close()
#     print("Consumer closed.")




from confluent_kafka import Consumer
import json

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "test"

config = {
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id": "mygroup1",
    "auto.offset.reset": "latest"
}

consumer = Consumer(config)
consumer.subscribe([TOPIC])

print("📥 Listening for Kafka messages...")

try:
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue

        if msg.error():
            print("Consumer error:", msg.error())
            continue

        data = json.loads(msg.value().decode("utf-8"))
        print("RECEIVED FLOW:", data)

except KeyboardInterrupt:
    print("Stopping consumer...")

consumer.close()
