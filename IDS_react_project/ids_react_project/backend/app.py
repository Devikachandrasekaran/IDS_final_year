# from flask import Flask
# from routes.prediction import prediction_bp
# from routes.dashboard import dashboard_bp

# def create_app():
#     app = Flask(__name__)

#     # Register blueprints
#     app.register_blueprint(prediction_bp, url_prefix="/api/prediction")
#     app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

#     return app

# if __name__ == "__main__":
#     app = create_app()
#     app.run(host="0.0.0.0", port=5000, debug=True)


# backend/app.py
# from flask import Flask
# from flask_cors import CORS
# from routes.prediction import prediction_bp
# from routes.dashboard import dashboard_bp

# def create_app():
#     app = Flask(__name__)

#     # ✅ Enable CORS for all origins and routes
#     CORS(app, resources={r"/*": {"origins": "*"}})

#     # Register blueprints
#     app.register_blueprint(prediction_bp, url_prefix="/api/prediction")
#     app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

#     @app.route("/")
#     def home():
#         return "Flask Backend Running OK ✅"

#     return app

# if __name__ == "__main__":
#     app = create_app()
#     app.run(host="0.0.0.0", port=5000, debug=True)


# backend/app.py
# from flask import Flask
# from flask_cors import CORS

# # Import blueprints
# from routes.prediction import prediction_bp
# from routes.dashboard import dashboard_bp

# app = Flask(__name__)
# CORS(app, origins=["http://localhost:3000"])  # allow React frontend

# # Register blueprints
# app.register_blueprint(prediction_bp, url_prefix="/api/prediction")
# app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)





from flask import Flask
from flask_cors import CORS

# Import Blueprints
from routes.prediction import prediction_bp
from routes.dashboard import dashboard_bp

def create_app():
    app = Flask(__name__)

    # Allow CORS for React frontend
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register API routes
    app.register_blueprint(prediction_bp, url_prefix="/api/prediction")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
