# backend/stats_store.py
# This will hold live stats for dashboard
# stats = {
#     "total_flows": 0,
#     "malicious_flows": 0
# }

# def update_stats(pred_label):
#     stats["total_flows"] += 1
#     if pred_label == 1:
#         stats["malicious_flows"] += 1

# def get_stats():
#     total = stats["total_flows"]
#     malicious = stats["malicious_flows"]
#     risk = f"{(malicious / total * 100):.1f}%" if total > 0 else "0%"
#     return {"total_flows": total, "malicious_flows": malicious, "risk": risk}



# backend/stats_store.py


# from flask import Flask, jsonify
# from flask_cors import CORS
# import stats_store

# app = Flask(__name__)
# CORS(app)

# @app.route('/api/prediction/live', methods=['GET'])
# def live_prediction_data():
#     data = stats_store.get_live_data()
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


# stats_store.py
# """
# This module stores and manages live prediction statistics.
# It is imported by both consumer_predict.py (for updates)
# and Flask app.py (for reading live data).
# """

# from threading import Lock

# # Thread-safe access lock
# _lock = Lock()

# # Global statistics dictionary
# _live_stats = {
#     "total_packets": 0,
#     "malicious": 0,
#     "benign": 0,
#     "latest_flows": []
# }


# def update_stats(flow: str, prediction: int):
#     """
#     Updates live statistics based on the new prediction result.

#     Args:
#         flow (str): Flow identifier (e.g., '192.168.1.1 -> 10.0.0.2')
#         prediction (int): 1 for malicious, 0 for benign
#     """
#     with _lock:
#         _live_stats["total_packets"] += 1
#         if prediction == 1:
#             _live_stats["malicious"] += 1
#         else:
#             _live_stats["benign"] += 1

#         # Append latest flow prediction
#         _live_stats["latest_flows"].append({
#             "flow": flow,
#             "prediction": prediction
#         })

#         # Keep only the latest 10 entries
#         if len(_live_stats["latest_flows"]) > 10:
#             _live_stats["latest_flows"].pop(0)


# def get_live_data():
#     """
#     Returns the current live statistics dictionary.
#     """
#     with _lock:
#         # Return a shallow copy to avoid threading issues
#         return dict(_live_stats)


# def reset_stats():
#     """
#     Resets all statistics to initial state.
#     Useful for restarting live prediction.
#     """
#     with _lock:
#         _live_stats["total_packets"] = 0
#         _live_stats["malicious"] = 0
#         _live_stats["benign"] = 0
#         _live_stats["latest_flows"] = []





# """
# This module stores and manages live prediction statistics.
# Thread-safe access for live dashboard.
# """

# from threading import Lock
# from collections import deque

# # Thread-safe lock
# _lock = Lock()

# # Live statistics
# _live_stats = {
#     "total_packets": 0,
#     "malicious": 0,
#     "benign": 0,
#     "latest_flows": deque(maxlen=20)  # store last 20 flows
# }

# def update_stats(flow: dict, prediction: int):
#     """
#     Updates live statistics with new flow prediction.
#     Args:
#         flow (dict): Flow dictionary containing src_ip and dst_ip.
#         prediction (int): 1 for malicious, 0 for benign
#     """
#     with _lock:
#         _live_stats["total_packets"] += 1
#         if prediction == 1:
#             _live_stats["malicious"] += 1
#         else:
#             _live_stats["benign"] += 1

#         # Add latest flow
#         _live_stats["latest_flows"].append({
#             "src_ip": flow.get("src_ip", ""),
#             "dst_ip": flow.get("dst_ip", ""),
#             "pred_label": prediction
#         })

# def get_live_data():
#     """
#     Returns current live statistics for dashboard and frontend.
#     """
#     with _lock:
#         total = _live_stats["total_packets"]
#         malicious = _live_stats["malicious"]
#         risk = int((malicious / total) * 100) if total > 0 else 0

#         return {
#             "total_packets": total,
#             "malicious_flows": malicious,
#             "benign_flows": _live_stats["benign"],
#             "risk_percent": risk,
#             "latest_flows": list(_live_stats["latest_flows"])
#         }

# def reset_stats():
#     """
#     Resets all stats.
#     """
#     with _lock:
#         _live_stats["total_packets"] = 0
#         _live_stats["malicious"] = 0
#         _live_stats["benign"] = 0
#         _live_stats["latest_flows"].clear()





# backend/utils/stats_store.py
from threading import Lock
from collections import deque

_lock = Lock()

_live_stats = {
    "total_packets": 0,
    "malicious": 0,
    "benign": 0,
    "latest_flows": deque(maxlen=50)  # store last N flows
}

def update_stats(flow: dict, prediction: int):
    with _lock:
        _live_stats["total_packets"] += 1
        if prediction == 1:
            _live_stats["malicious"] += 1
        else:
            _live_stats["benign"] += 1

        _live_stats["latest_flows"].appendleft({
            "src_ip": flow.get("src_ip", ""),
            "dst_ip": flow.get("dst_ip", ""),
            "pred_label": int(prediction),
            "ts": flow.get("ts", None)
        })

def get_live_data():
    with _lock:
        total = _live_stats["total_packets"]
        malicious = _live_stats["malicious"]
        risk = int((malicious / total) * 100) if total > 0 else 0

        return {
            "total_packets": total,
            "malicious_flows": malicious,
            "benign_flows": _live_stats["benign"],
            "risk_percent": risk,
            "latest_flows": list(_live_stats["latest_flows"])
        }

def reset_stats():
    with _lock:
        _live_stats["total_packets"] = 0
        _live_stats["malicious"] = 0
        _live_stats["benign"] = 0
        _live_stats["latest_flows"].clear()
