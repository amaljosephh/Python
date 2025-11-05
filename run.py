from app import create_app
from app.routes import initialize_product_vectors

app = create_app()

with app.app_context():
    initialize_product_vectors()

if __name__ == '__main__':
    app.run(debug=True) 