from flask import Flask, render_template, request, redirect, url_for, flash, session, abort, send_from_directory
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv
from database import db, User, AnalysisResult
from pipeline.inference import SoilCrackPipeline
from PIL import Image
import numpy as np
import cloudinary
import cloudinary.uploader
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key-change-me')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MASK_FOLDER'] = 'uploads/masks'
app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'development')

# Cloudinary Configuration
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

pipeline = SoilCrackPipeline()

# Ensure folders exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Admin Required Decorator
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        user = User(username=username, email=email, 
                    password_hash=generate_password_hash(password))
        
        # First registered user could be admin or use manual SQL as requested
        # For now, following instructions to leave role as default 'user'
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/history')
@login_required
def history():
    analyses = AnalysisResult.query.filter_by(user_id=current_user.id).order_by(AnalysisResult.created_at.desc()).all()
    return render_template('history.html', history=analyses)

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    users = User.query.all()
    analyses = AnalysisResult.query.all()
    return render_template('admin_dashboard.html', 
                           users=users, 
                           analyses=analyses,
                           total_users=len(users),
                           total_analyses=len(analyses))

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == 'admin':
        flash('Cannot delete admin user')
    else:
        # Delete from Cloudinary first if IDs exist
        if user.analyses:
            for analysis in user.analyses:
                if analysis.original_public_id:
                    cloudinary.uploader.destroy(analysis.original_public_id)
                if analysis.mask_public_id:
                    cloudinary.uploader.destroy(analysis.mask_public_id)
                if analysis.highlight_public_id:
                    cloudinary.uploader.destroy(analysis.highlight_public_id)
        
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} deleted')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_analysis/<int:analysis_id>', methods=['POST'])
@login_required
@admin_required
def delete_analysis(analysis_id):
    analysis = AnalysisResult.query.get_or_404(analysis_id)
    
    # Delete from Cloudinary
    if analysis.original_public_id:
        cloudinary.uploader.destroy(analysis.original_public_id)
    if analysis.mask_public_id:
        cloudinary.uploader.destroy(analysis.mask_public_id)
    if analysis.highlight_public_id:
        cloudinary.uploader.destroy(analysis.highlight_public_id)
        
    db.session.delete(analysis)
    db.session.commit()
    flash('Analysis record deleted')
    return redirect(url_for('admin_dashboard'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/comparison')
def comparison():
    return render_template('comparison.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # In-memory upload of original image
    upload_result = cloudinary.uploader.upload(file, folder="soil_crack/original")
    original_url = upload_result['secure_url']
    original_public_id = upload_result['public_id']

    # For processing, we still need a local copy or convert file to something PIL can use
    # Since we don't want to modify pipeline, let's keep a temporary local save for inference
    # or pass the file stream if pipeline supports it. SoilCrackPipeline expects a path.
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.seek(0) # Reset file pointer after Cloudinary upload
    file.save(temp_path)

    result = pipeline.run(temp_path)

    if result['crack_found']:
        # Upload Mask
        mask_io = io.BytesIO()
        mask_img = Image.fromarray((result['mask'] * 255).astype(np.uint8))
        mask_img.save(mask_io, format='PNG')
        mask_io.seek(0)
        mask_upload = cloudinary.uploader.upload(mask_io, folder="soil_crack/mask")
        mask_url = mask_upload['secure_url']
        mask_public_id = mask_upload['public_id']
        
        # Upload Highlighted Image
        highlight_io = io.BytesIO()
        result['highlight'].save(highlight_io, format='PNG')
        highlight_io.seek(0)
        highlight_upload = cloudinary.uploader.upload(highlight_io, folder="soil_crack/highlight")
        highlight_url = highlight_upload['secure_url']
        highlight_public_id = highlight_upload['public_id']

        analysis = AnalysisResult(
            user_id=current_user.id,
            original_image_url=original_url,
            original_public_id=original_public_id,
            mask_image_url=mask_url,
            mask_public_id=mask_public_id,
            highlight_image_url=highlight_url,
            highlight_public_id=highlight_public_id,
            crack_length=result['length'],
            crack_width=result['width'],
            crack_area=result['area'],
            severity=result['severity'],
            recommendation="\n".join(result['recommendation'])
        )
        db.session.add(analysis)
        db.session.commit()
        
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return render_template('result.html', result=result, 
                               original_url=original_url, 
                               mask_url=mask_url, 
                               highlight_url=highlight_url)
    else:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return render_template('result.html', result=result, original_url=original_url)

if __name__ == '__main__':
    with app.app_context():
        # Using db.create_all() for development. In production, migrations are preferred.
        db.create_all()
    app.run(debug=(app.config['FLASK_ENV'] == 'development'))
