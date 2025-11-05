import os
from flask import Blueprint, render_template, request, current_app, url_for
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
import torch
import json

main = Blueprint('main', __name__)

# Load the CLIP model for image feature extraction
model = SentenceTransformer('clip-ViT-B-32')

# Product database (in a real application, this would be in a database)
PRODUCTS = [
    {
        'id': 1,
        'name': 'Cotton-Jersey Zip-Up Hoodie',
        'brand': 'adidas',
        'price': 1450,
        'image_path': 'img/f1.jpg',
        'vector': None,
        'description': 'A comfortable cotton-jersey zip-up hoodie perfect for casual wear'
    },
    {
        'id': 2,
        'name': 'Casual Cotton T-Shirt',
        'brand': 'Nike',
        'price': 1250,
        'image_path': 'img/f2.jpg',
        'vector': None,
        'description': 'Classic cotton t-shirt with modern design and comfortable fit'
    },
    {
        'id': 3,
        'name': 'Printed Summer Shirt',
        'brand': 'Puma',
        'price': 1350,
        'image_path': 'img/f3.jpg',
        'vector': None,
        'description': 'Vibrant printed summer shirt perfect for beach and casual outings'
    },
    {
        'id': 4,
        'name': 'Floral Pattern Dress',
        'brand': 'Zara',
        'price': 1550,
        'image_path': 'img/f4.jpg',
        'vector': None,
        'description': 'Beautiful floral pattern dress ideal for summer occasions'
    },
    {
        'id': 5,
        'name': 'Denim Jacket',
        'brand': 'Levis',
        'price': 2450,
        'image_path': 'img/f5.jpg',
        'vector': None,
        'description': 'Classic denim jacket with modern styling and perfect fit'
    },
    {
        'id': 6,
        'name': 'Classic White Shirt',
        'brand': 'H&M',
        'price': 950,
        'image_path': 'img/f6.jpg',
        'vector': None,
        'description': 'Timeless white shirt suitable for both casual and formal occasions'
    },
    {
        'id': 7,
        'name': 'Summer Beach Shirt',
        'brand': 'Pull&Bear',
        'price': 1150,
        'image_path': 'img/f7.jpg',
        'vector': None,
        'description': 'Light and breezy summer shirt with tropical patterns'
    },
    {
        'id': 8,
        'name': 'Vintage Denim Shirt',
        'brand': 'Gap',
        'price': 1650,
        'image_path': 'img/f8.jpg',
        'vector': None,
        'description': 'Vintage-style denim shirt with classic American design'
    }
]

def encode_image(image_path):
    """Encode image using CLIP model."""
    return model.encode(Image.open(image_path))

def compute_similarity(query_vector, product_vector):
    """Compute cosine similarity between two vectors using pytorch."""
    return util.cos_sim(
        torch.tensor(query_vector).unsqueeze(0),
        torch.tensor(product_vector).unsqueeze(0)
    ).item()

def initialize_product_vectors():
    """Initialize product vectors on startup."""
    print("Initializing product vectors...")
    for product in PRODUCTS:
        image_path = os.path.join(current_app.static_folder, product['image_path'])
        if os.path.exists(image_path):
            try:
                # Encode both image and text description
                img_embedding = encode_image(image_path)
                text_embedding = model.encode(product['description'])
                # Combine image and text embeddings (average them)
                product['vector'] = (img_embedding + text_embedding) / 2
                print(f"Encoded {product['name']}")
            except Exception as e:
                print(f"Error encoding {product['name']}: {str(e)}")
                product['vector'] = None
        else:
            print(f"Image not found: {image_path}")
    print("Initialization complete!")

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/products')
def products():
    return render_template('products.html', products=PRODUCTS)

@main.route('/search_products', methods=['POST'])
def search_products():
    if 'image' not in request.files:
        return render_template('products.html', products=PRODUCTS, error="No image uploaded")
    
    file = request.files['image']
    if file.filename == '':
        return render_template('products.html', products=PRODUCTS, error="No image selected")
    
    if file:
        try:
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join(current_app.static_folder, 'temp', filename)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            file.save(temp_path)
            
            # Convert image to vector using CLIP
            query_vector = encode_image(temp_path)
            
            # Remove temporary file
            os.remove(temp_path)
            
            # Calculate similarities and sort products
            similarities = []
            for product in PRODUCTS:
                if product['vector'] is not None:
                    similarity = compute_similarity(query_vector, product['vector'])
                    similarities.append((similarity, product))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[0], reverse=True)
            sorted_products = [item[1] for item in similarities]
            
            return render_template('products.html', products=sorted_products)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return render_template('products.html', products=PRODUCTS, error="Error processing image")

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/blog')
def blog():
    return render_template('blog.html')

@main.route('/contact')
def contact():
    return render_template('contact.html') 