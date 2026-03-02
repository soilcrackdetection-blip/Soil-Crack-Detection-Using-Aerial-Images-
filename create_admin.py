from app import app

from database import db, User
from werkzeug.security import generate_password_hash
import os
from dotenv import load_dotenv

load_dotenv()

admin_email = os.getenv('ADMIN_EMAIL')
if not admin_email:
    raise ValueError('ADMIN_EMAIL not set in .env')

admin = User(
    username='admin',
    email=admin_email,
    password_hash=generate_password_hash('ChangeMe123!'),  # TODO: change password
    role='admin'
)

def create_admin():
    with app.app_context():
        db.session.add(admin)
        db.session.commit()
        print(f'Admin user created: {admin_email}')

if __name__ == '__main__':
    create_admin()
