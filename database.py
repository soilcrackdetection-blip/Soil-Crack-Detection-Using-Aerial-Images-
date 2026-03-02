from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user') # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete="CASCADE"), nullable=False)
    # Cloudinary fields
    original_image_url = db.Column(db.String, nullable=False)
    original_public_id = db.Column(db.String)
    mask_image_url = db.Column(db.String)
    mask_public_id = db.Column(db.String)
    highlight_image_url = db.Column(db.String)
    highlight_public_id = db.Column(db.String)
    
    crack_length = db.Column(db.Float)
    crack_width = db.Column(db.Float)
    crack_area = db.Column(db.Float)
    severity = db.Column(db.String(50))
    recommendation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('analyses', lazy=True, cascade="all, delete-orphan"))
