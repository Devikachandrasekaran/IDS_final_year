# from flask import Blueprint, jsonify
# from utils.kafka_consumer import get_dashboard_summary

# dashboard_bp = Blueprint("dashboard_bp", __name__)

# @dashboard_bp.route("/", methods=["GET"])
# def dashboard():
#     """
#     Returns a small summary for dashboard:
#     { total_flows, malicious_flows, risk_percent }
#     """
#     summary = get_dashboard_summary()
#     return jsonify(summary)



# from flask import Blueprint, jsonify
# from stats_store import get_stats

# dashboard_bp = Blueprint("dashboard", __name__)

# @dashboard_bp.route("/api/dashboard", methods=["GET"])
# def dashboard_stats():
#     return jsonify(get_stats())


# backend/routes/dashboard.py
# from flask import Blueprint, jsonify
# from stats_store import get_stats

# dashboard_bp = Blueprint("dashboard_bp", __name__)

# @dashboard_bp.route("/", methods=["GET"])
# def dashboard():
#     return jsonify(get_stats())




# # backend/routes/dashboard.py
# from flask import Blueprint, jsonify
# from stats_store import get_live_data

# dashboard_bp = Blueprint("dashboard_bp", __name__)

# @dashboard_bp.route("/", methods=["GET"])
# def dashboard():
#     """
#     Returns current live statistics for the dashboard.
#     Structure:
#     {
#         total_packets,
#         malicious,
#         benign,
#         risk_percent,
#         latest_flows: [...]
#     }
#     """
#     data = get_live_data()

#     # Calculate risk percentage safely
#     total = data.get("total_packets", 0)
#     malicious = data.get("malicious", 0)
#     risk_percent = round((malicious / total) * 100, 2) if total > 0 else 0

#     # Prepare final response
#     result = {
#         "total_packets": total,
#         "malicious_flows": malicious,
#         "benign_flows": data.get("benign", 0),
#         "risk_percent": risk_percent,
#         "latest_flows": data.get("latest_flows", [])
#     }

#     return jsonify(result)




# from flask import Blueprint, jsonify
# from utils.stats_store import get_live_data

# dashboard_bp = Blueprint("dashboard_bp", __name__)

# @dashboard_bp.route("/", methods=["GET"])
# def dashboard():
#     """
#     Returns total packets, malicious flows, and risk %.
#     """
#     data = get_live_data()
#     summary = {
#         "total_packets": data["total_packets"],
#         "malicious_flows": data["malicious_flows"],
#         "risk_percent": data["risk_percent"]
#     }
#     return jsonify(summary)


# # backend/routes/dashboard.py
# from flask import Blueprint, jsonify
# from utils.stats_store import get_live_data

# dashboard_bp = Blueprint("dashboard_bp", __name__)

# @dashboard_bp.route("/", methods=["GET"])
# def dashboard_summary():
#     """
#     Returns dashboard summary: total packets, malicious packets, risk percentage.
#     """
#     stats = get_live_data()
#     total = stats.get("total_packets", 0)
#     malicious = stats.get("malicious", 0)
#     risk_percent = round((malicious / total) * 100, 2) if total > 0 else 0

#     summary = {
#         "total_packets": total,
#         "malicious_flows": malicious,
#         "risk_percent": risk_percent
#     }
#     return jsonify(summary)




# backend/routes/dashboard.py
from flask import Blueprint, jsonify
from utils.stats_store import get_live_data

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/summary", methods=["GET"])
def summary():
    data = get_live_data()
    return jsonify(data)
