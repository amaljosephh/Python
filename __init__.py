from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configuration settings
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    
    # Register blueprints
    from .routes import main
    app.register_blueprint(main)
    
    return app 