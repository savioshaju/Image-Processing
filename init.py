import os
from flask import Flask
from config import Config
from supabase import create_client

supabase = None  # global reference


def create_app(config_class=Config):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class)

    # Initialize Supabase client once
    global supabase
    supabase = create_client(app.config["SUPABASE_URL"], app.config["SUPABASE_KEY"])
    app.supabase = supabase

    # Register blueprints
    from routes import main_bp
    app.register_blueprint(main_bp)

    return app
