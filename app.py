import os
import io
import requests
from flask import send_file  # Import send_file
from flask import Flask, render_template, redirect, request, jsonify, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from models import db
from models import Post
from models import Comment
from models import User
from flask_bcrypt import Bcrypt
from flask_mail import Message
import pyotp
import random
import string
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import CNN  # Ensure CNN.py exists
from download_model import download_model
download_model()

import torch
from CNN import CNN, idx_to_classes  # Import your model class and classes

# Number of classes
K = len(idx_to_classes)

# Initialize the model architecture
model = CNN(K)

# Load the state dict (weights only)
state_dict = torch.load("plant_disease_model_1_latest.pt", map_location="cpu")

# Apply weights to the model
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

print("Model loaded successfully!")



# Load Disease and Supplement Data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')


# Initialize Flask App
app = Flask(__name__)

# Set Absolute Path for Uploads
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# OpenWeatherMap API Key
API_KEY = "09aa1e51554fd886885bd42685bb82f2"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

def prediction(image_path):
    """ Predict disease from image """
    image = Image.open(image_path).resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    output = model(input_data).detach().numpy()
    return np.argmax(output)

def predict_disease_risk(humidity, temperature):
    """ Predict disease risk based on weather conditions """
    if humidity > 80 and temperature > 25:
        return "🔴 High Risk of Fungal Diseases (Avoid Overwatering)"
    elif humidity > 60 and temperature > 20:
        return "🟡 Moderate Risk of Disease"
    else:
        return "🟢 Low Risk (Weather is Good for Plants)"

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/customer')
def customer():
    return render_template('customer-forum.html')

medicines = [
        {"name": "Organic Potting Soil", "price": 100},
        {"name": "Plant Growth Booster", "price": 80},
        {"name": "Neem Oil Spray", "price": 120},
        {"name": "Insect Sticky Traps", "price": 90},
        {"name": "Garden Gloves", "price": 90},
        {"name": "Trellis", "price": 90},
        {"name": "pH & Moisture Meter ", "price": 90},
        {"name": "Compost", "price": 90},

    ]

@app.route('/Ecommerce')
def Ecommerce():
    return render_template('Ecommerce.html', medicines=medicines)

@app.route("/add_to_cart", methods=["POST"])
def add_to_cart():
    name = request.form.get("name")
    price = float(request.form.get("price"))

    if "cart" not in session:
        session["cart"] = []

    cart = session["cart"]

    # Check if the item is already in the cart
    for item in cart:
        if item["name"] == name:
            item["quantity"] += 1
            break
    else:
        cart.append({"name": name, "price": price, "quantity": 1})

    session["cart"] = cart

    # Calculate totals
    subtotal = sum(item["price"] * item["quantity"] for item in cart)
    gst = subtotal * 0.18
    discount = subtotal * 0.1 if subtotal > 500 else 0
    total = subtotal + gst - discount

    # Store in session
    session["subtotal"] = round(subtotal, 2)
    session["gst"] = round(gst, 2)
    session["discount"] = round(discount, 2)
    session["total"] = round(total, 2)

    session.modified = True  # Ensure session updates persist

    # Debugging: Print session data in the console
    print("Cart:", session["cart"])
    print("Subtotal:", session["subtotal"])
    print("GST:", session["gst"])
    print("Discount:", session["discount"])
    print("Total:", session["total"])

    return redirect(url_for("Ecommerce"))


@app.route('/checkout')
def checkout():
    cart = session.get('cart', [])
    subtotal = sum(item['price'] * item['quantity'] for item in cart)
    gst = subtotal * 0.18
    discount = 50 if subtotal > 500 else 0
    total = subtotal + gst - discount

    return render_template('checkout.html', cart=cart, subtotal=subtotal, gst=gst, discount=discount, total=total)


@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    session['cart'] = []  # Empty the cart
    session['subtotal'] = 0
    session['gst'] = 0
    session['discount'] = 0
    session['total'] = 0
    session.modified = True  # Ensure Flask updates session
    return redirect(url_for('medicine_order'))  # Redirect to order page



app.config['SECRET_KEY'] = 'a3f1b9e2d6c8471f9b23e5ad6f2a4b7d1c9e6a2d8b3f5c7e1d4f2a1b9e8c3d7f'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///forum.db'
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
with app.app_context():
    db.create_all()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('customer_forum'))
        else:
            flash('Login unsuccessful. Check email and password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Removes user session
    return redirect(url_for('login'))  # Redirects to login page



@app.route('/customer')
def customer_forum():
    posts = Post.query.order_by(Post.date_posted.desc()).all()
    return render_template('customer-forum.html', posts=posts)

@app.route('/create_post', methods=['GET', 'POST'])
@login_required
def create_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        post = Post(title=title, content=content, user_id=current_user.id)
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('customer_forum'))
    return render_template('create_post.html')

@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def post(post_id):
    post = Post.query.get_or_404(post_id)
    comments = Comment.query.filter_by(post_id=post.id).all()
    if request.method == 'POST':
        if not current_user.is_authenticated:
            flash("You need to log in to comment.", "danger")
            return redirect(url_for('login'))
        comment_content = request.form['content']
        comment = Comment(content=comment_content, post_id=post.id, user_id=current_user.id)
        db.session.add(comment)
        db.session.commit()
        return redirect(url_for('post', post_id=post.id))
    return render_template('post.html', post=post, comments=comments)


@app.route('/weather')
def get_weather():
    """ Fetch live weather data based on city, defaults to Delhi """
    city = request.args.get("city", "Delhi")  # Default city is Delhi

    url = f"{WEATHER_API_URL}?q={city}&appid={API_KEY}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return jsonify({"error": data.get("message", "Failed to fetch weather data")}), response.status_code

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        weather_condition = data["weather"][0]["description"]

        disease_risk = predict_disease_risk(humidity, temperature)

        weather_data = {
            "city": city,
            "temperature": f"{temperature}°C",
            "humidity": f"{humidity}%",
            "weather": weather_condition.capitalize(),
            "disease_risk": disease_risk
        }

        return jsonify(weather_data)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to connect to weather service"}), 500

@app.route('/weather-page')
def weather_page():
    return render_template("weather.html")



def get_last_conv_layer(model):
    """Find the last convolutional layer in a model, even if nested."""
    last_conv_layer = None

    for name, layer in model.named_modules():  # Use named_modules to go deep
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = name  # Keep updating with the last found Conv2D layer

    if last_conv_layer is None:
        raise ValueError("No convolutional layers found in the model! Check the architecture.")

    return last_conv_layer  # Return the last found Conv2D layer name



def generate_gradcam(image_path, model):
    """ Generate Grad-CAM heatmap using PIL """
    image = Image.open(image_path).resize((224, 224))
    input_tensor = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension

    # Select the last convolutional layer dynamically
    last_conv_layer_name = get_last_conv_layer(model)
    if last_conv_layer_name is None:
        raise ValueError("No convolutional layers found in the model!")

    conv_layer = dict(model.named_modules())[last_conv_layer_name]

    # Hook to capture gradients
    gradients = None
    activation = None

    def save_grad(grad):
        nonlocal gradients
        gradients = grad

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output

    conv_layer.register_forward_hook(forward_hook)
    handle = conv_layer.register_full_backward_hook(lambda m, g_in, g_out: save_grad(g_out[0]))

    # Forward Pass
    model.zero_grad()
    output = model(input_tensor)
    class_idx = output.argmax().item()

    # Backward Pass
    loss = output[0, class_idx]
    loss.backward()

    # Compute Grad-CAM heatmap
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activation.shape[1]):
        activation[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activation, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    # Convert to PIL Image
    heatmap = Image.fromarray(np.uint8(heatmap * 255)).convert("L")
    heatmap = heatmap.resize((224, 224), resample=Image.BICUBIC)

    # Overlay Heatmap
    original = image.resize((224, 224))
    heatmap = heatmap.convert("RGBA").resize(original.size)
    blended = Image.blend(original.convert("RGBA"), heatmap, alpha=0.5)

    return blended

@app.route('/submit', methods=['POST'])
def submit():
    """ Process uploaded plant image and return disease prediction with Grad-CAM """
    if 'image' not in request.files or request.files['image'].filename == '':
        return "No image uploaded!", 400  

    image = request.files['image']
    filename = image.filename  
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(file_path)

    # Make prediction
    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    # Prepare uploaded image URL for display
    uploaded_image_url = f"/static/uploads/{filename}"  # Adjust based on Flask static folder

    # Check if the leaf is diseased before generating Grad-CAM
    gradcam_url = None  # Default to None
    diseased_classes = [3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38]  # Healthy classes

    if pred not in diseased_classes:  
        gradcam_image = generate_gradcam(file_path, model)

        # Convert to RGB if RGBA (Fix OSError issue)
        if gradcam_image.mode == 'RGBA':
            gradcam_image = gradcam_image.convert('RGB')

        # Save Grad-CAM image in static folder
        gradcam_filename = f"gradcam_{filename}.jpg"  
        gradcam_path = os.path.join("static/gradcam", gradcam_filename)
        os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)
        gradcam_image.save(gradcam_path, format="JPEG")  # Ensure it's saved as JPEG

        gradcam_url = f"/static/gradcam/{gradcam_filename}"  # Correct path for rendering

    # Render submit.html and pass all required data
    return render_template('submit.html', 
                           title=title, 
                           desc=description, 
                           prevent=prevent, 
                           image_url=image_url,
                           simage=supplement_image_url,
                           sname=supplement_name,
                           buy_link=supplement_buy_link,
                           pred=pred,
                           uploaded_image_url=uploaded_image_url,  # Ensure uploaded image is always displayed
                           gradcam_url=gradcam_url  # Pass Grad-CAM URL (None if healthy)
                          )



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
