# # backend/utils/kafka_consumer.py
# import json
# import threading
# import time
# from collections import deque
# from confluent_kafka import Consumer, KafkaException
# from models.load_model import model, scaler, device
# from utils.preprocess import flow_to_feature_vector, make_tensor_from_vec
# import torch

# # Kafka & runtime config
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "test"            # ensure this matches your producer
# GROUP_ID = "ids-consumer-group"
# POLL_TIMEOUT = 1.0        # seconds

# # In-memory storage for recent predictions
# MAX_RECENT = 2000
# _recent = deque(maxlen=MAX_RECENT)
# _lock = threading.Lock()

# # Configure consumer
# _consumer = Consumer({
#     "bootstrap.servers": KAFKA_BOOTSTRAP,
#     "group.id": GROUP_ID,
#     "auto.offset.reset": "latest",   # start from newest messages
#     # optionally: "enable.auto.commit": False
# })
# _consumer.subscribe([TOPIC])

# def _predict_and_store(flow_json):
#     try:
#         vec = flow_to_feature_vector(flow_json)           # numpy vector shape (8,)
#         input_tensor = make_tensor_from_vec(vec, scaler, device)  # (1,1,feat)
#         with torch.no_grad():
#             prob, _ = model(input_tensor)
#             # model returns tensor shape (1,)
#             # if model was saved differently, adapt accordingly
#             p = float(prob.cpu().numpy().ravel()[0])
#             label = int(p > 0.5)
#     except Exception as e:
#         # on error, store as unknown
#         p = 0.0
#         label = 0
#     entry = {
#         "src_ip": flow_json.get("src_ip"),
#         "dst_ip": flow_json.get("dst_ip"),
#         "src_port": flow_json.get("src_port"),
#         "dst_port": flow_json.get("dst_port"),
#         "protocol": flow_json.get("protocol"),
#         "pkt_count": flow_json.get("pkt_count"),
#         "byte_count": flow_json.get("byte_count"),
#         "duration": flow_json.get("duration"),
#         "pred": label,
#         "prob": round(p, 4),
#         "ts": time.time()
#     }
#     with _lock:
#         _recent.append(entry)

# def _consumer_loop():
#     """
#     Background loop that polls Kafka for new flow messages,
#     predicts and keeps recent results.
#     """
#     while True:
#         try:
#             msg = _consumer.poll(POLL_TIMEOUT)
#             if msg is None:
#                 continue
#             if msg.error():
#                 # handle error (log)
#                 # print("Kafka error:", msg.error())
#                 continue
#             # parse JSON message
#             try:
#                 flow_json = json.loads(msg.value().decode("utf-8"))
#             except Exception:
#                 continue
#             _predict_and_store(flow_json)
#         except KafkaException as ke:
#             # handle or log
#             # print("Kafka exception:", ke)
#             time.sleep(1)
#         except Exception:
#             time.sleep(1)

# # Start consumer in background thread (daemon)
# _thread = threading.Thread(target=_consumer_loop, daemon=True)
# _thread.start()

# # Public helpers for API routes
# def get_recent_predictions(limit=20):
#     with _lock:
#         items = list(_recent)[-limit:]
#     # return newest-first
#     return items[::-1]

# def get_dashboard_summary():
#     with _lock:
#         total = len(_recent)
#         malicious = sum(1 for i in _recent if i.get("pred") == 1)
#     risk = (malicious / total * 100) if total else 0.0
#     return {
#         "total_flows": total,
#         "malicious_flows": malicious,
#         "risk": f"{risk:.2f}%"
#     }



# import json
# import threading
# import time
# from collections import deque
# from confluent_kafka import Consumer, KafkaException
# import torch


# # from backend.models.load_model import model, scaler, device
# # from backend.utils.preprocess import flow_to_feature_vector, make_tensor_from_vec


# from models.load_model import model, scaler, device
# from utils.preprocess import flow_to_feature_vector, make_tensor_from_vec


# # from models.load_model import model, scaler, device
# # from utils.preprocess import flow_to_feature_vector, make_tensor_from_vec

# # Kafka config
# KAFKA_BOOTSTRAP = "localhost:9092"
# TOPIC = "test"
# GROUP_ID = "ids-consumer-group"
# POLL_TIMEOUT = 1.0

# # In-memory storage
# MAX_RECENT = 2000
# _recent = deque(maxlen=MAX_RECENT)
# _lock = threading.Lock()

# # Configure consumer
# _consumer = Consumer({
#     "bootstrap.servers": KAFKA_BOOTSTRAP,
#     "group.id": GROUP_ID,
#     "auto.offset.reset": "latest",
# })
# _consumer.subscribe([TOPIC])

# # Predict & store
# def _predict_and_store(flow_json):
#     try:
#         vec = flow_to_feature_vector(flow_json)
#         tensor = make_tensor_from_vec(vec, scaler, device)
#         with torch.no_grad():
#             prob, _ = model(tensor)
#             p = float(prob.cpu().numpy().ravel()[0])
#             label = int(p > 0.5)
#     except Exception:
#         p, label = 0.0, 0

#     entry = {
#         "src_ip": flow_json.get("src_ip"),
#         "dst_ip": flow_json.get("dst_ip"),
#         "src_port": flow_json.get("src_port"),
#         "dst_port": flow_json.get("dst_port"),
#         "protocol": flow_json.get("protocol"),
#         "pkt_count": flow_json.get("pkt_count"),
#         "byte_count": flow_json.get("byte_count"),
#         "duration": flow_json.get("duration"),
#         "pred": label,
#         "prob": round(p, 4),
#         "ts": time.time()
#     }
#     with _lock:
#         _recent.append(entry)

# # Background consumer loop
# def _consumer_loop():
#     while True:
#         try:
#             msg = _consumer.poll(POLL_TIMEOUT)
#             if msg is None:
#                 continue
#             if msg.error():
#                 continue
#             try:
#                 flow_json = json.loads(msg.value().decode("utf-8"))
#             except Exception:
#                 continue
#             _predict_and_store(flow_json)
#         except KafkaException:
#             time.sleep(1)
#         except Exception:
#             time.sleep(1)

# # Start background thread
# _thread = threading.Thread(target=_consumer_loop, daemon=True)
# _thread.start()

# # Public API
# def get_recent_predictions(limit=20):
#     with _lock:
#         items = list(_recent)[-limit:]
#     return items[::-1]

# def get_dashboard_summary():
#     with _lock:
#         total = len(_recent)
#         malicious = sum(1 for i in _recent if i.get("pred") == 1)
#     risk = (malicious / total * 100) if total else 0.0
#     return {
#         "total_flows": total,
#         "malicious_flows": malicious,
#         "risk": f"{risk:.2f}%"
#     }





# backend/utils/kafka_consumer.py
"""
Alternative background consumer which collects recent predictions in memory.
Run as module from backend dir if you want:
    python -m utils.kafka_consumer
This script demonstrates an in-memory deque of recent predictions and exposes
functions that other modules can import:
    from utils.kafka_consumer import get_recent_predictions, get_dashboard_summary
"""

import json
import threading
import time
from collections import deque
from confluent_kafka import Consumer, KafkaException
import torch
from models.load_model import model, scaler, device
from utils.preprocess import flow_to_feature_vector, make_tensor_from_vec

# Kafka config
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "test"
GROUP_ID = "ids-consumer-group"
POLL_TIMEOUT = 1.0

# In-memory storage
MAX_RECENT = 2000
_recent = deque(maxlen=MAX_RECENT)
_lock = threading.Lock()

_consumer = Consumer({
    "bootstrap.servers": KAFKA_BOOTSTRAP,
    "group.id": GROUP_ID,
    "auto.offset.reset": "latest",
})
_consumer.subscribe([TOPIC])

def _predict_and_store(flow_json):
    try:
        vec = flow_to_feature_vector(flow_json)
        tensor = make_tensor_from_vec(vec, scaler, device)
        with torch.no_grad():
            prob_tensor, _ = model(tensor)
            prob = float(prob_tensor.cpu().numpy().ravel()[0])
            label = int(prob > 0.5)
    except Exception as e:
        print("Prediction error in background consumer:", e)
        label, prob = 0, 0.0

    entry = {
        "src_ip": flow_json.get("src_ip"),
        "dst_ip": flow_json.get("dst_ip"),
        "src_port": flow_json.get("src_port"),
        "dst_port": flow_json.get("dst_port"),
        "protocol": flow_json.get("protocol"),
        "pkt_count": flow_json.get("pkt_count"),
        "byte_count": flow_json.get("byte_count"),
        "duration": flow_json.get("duration"),
        "pred": label,
        "prob": round(prob, 4),
        "ts": time.time()
    }
    with _lock:
        _recent.append(entry)

def _consumer_loop():
    while True:
        try:
            msg = _consumer.poll(POLL_TIMEOUT)
            if msg is None:
                continue
            if msg.error():
                print("Consumer error:", msg.error())
                continue
            try:
                raw = msg.value()
                s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                flow_json = json.loads(s)
            except Exception as e:
                print("Skipping malformed message:", e)
                continue
            _predict_and_store(flow_json)
        except KafkaException:
            time.sleep(1)
        except Exception as e:
            print("Background consumer unexpected:", e)
            time.sleep(1)

# Start background thread on import (daemon so process can exit)
_thread = threading.Thread(target=_consumer_loop, daemon=True)
_thread.start()

def get_recent_predictions(limit=20):
    with _lock:
        items = list(_recent)[-limit:]
    return items[::-1]

def get_dashboard_summary():
    with _lock:
        total = len(_recent)
        malicious = sum(1 for i in _recent if i.get("pred") == 1)
    risk = (malicious / total * 100) if total else 0.0
    return {
        "total_flows": total,
        "malicious_flows": malicious,
        "risk": f"{risk:.2f}%"
    }

if __name__ == "__main__":
    print("Background consumer started, collecting predictions in memory. Ctrl-C to exit.")
    try:
        while True:
            time.sleep(2)
            print("Summary:", get_dashboard_summary())
    except KeyboardInterrupt:
        print("Exiting background consumer.")
        _consumer.close()
