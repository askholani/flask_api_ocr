# api/__init__.py
from flask import Flask
from firebase_admin import credentials, initialize_app

crud = credentials.Certificate("api/key.json")
default_app = initialize_app(crud)

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = '12345rtfescdvf'

    from .userApi import userAPI  # Correct Blueprint name

    app.register_blueprint(userAPI, url_prefix='/products')  # Correct URL prefix
    return app
