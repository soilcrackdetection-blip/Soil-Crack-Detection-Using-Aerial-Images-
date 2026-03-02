from app import app
from database import db

def init():
    """Create all tables in the Neon PostgreSQL database using Flask app context."""
    with app.app_context():
        print("Dropping all tables...")
        db.drop_all()
        print("Creating all tables...")
        db.create_all()
    print("Database tables created.")

if __name__ == "__main__":
    init()
