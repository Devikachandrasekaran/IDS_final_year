# from flask import Blueprint, jsonify
# from utils.kafka_consumer import get_recent_predictions

# prediction_bp = Blueprint("prediction_bp", __name__)

# @prediction_bp.route("/live", methods=["GET"])
# def live_predictions():
#     """
#     Returns the most recent flow predictions (JSON array).
#     Frontend polls this endpoint periodically when "Start Prediction" is clicked.
#     """
#     data = get_recent_predictions(limit=20)
#     return jsonify(data)


# from flask import Blueprint, jsonify
# from utils.stats_store import get_live_data

# prediction_bp = Blueprint("prediction_bp", __name__)

# @prediction_bp.route("/live", methods=["GET"])
# def live_predictions():
#     """
#     Returns recent flows for frontend Home.
#     """
#     data = get_live_data()
#     return jsonify(data["latest_flows"])


# backend/routes/prediction.py
# from flask import Blueprint, jsonify
# from utils.stats_store import get_live_data

# prediction_bp = Blueprint("prediction_bp", __name__)

# @prediction_bp.route("/live", methods=["GET"])
# def live_predictions():
#     """
#     Returns the most recent flow predictions (JSON array).
#     Frontend polls this endpoint periodically when "Start Prediction" is clicked.
#     """
#     data = get_live_data().get("latest_flows", [])
#     return jsonify(data)




# backend/routes/prediction.py
from flask import Blueprint, jsonify
from utils.kafka_consumer import get_recent_predictions

prediction_bp = Blueprint("prediction", __name__)

@prediction_bp.route("/recent", methods=["GET"])
def recent():
    items = get_recent_predictions(limit=20)
    return jsonify(items)
