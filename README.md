# Soil Crack Detection Platform

Professional AI-driven platform for detecting and analyzing soil cracks using advanced segmentation and classification models.

## Features
- **AI-Powered Analysis**: Uses FCCN-based segmentation and classification.
- **Severity Scoring**: Multi-factor severity assessment with actionable recommendations.
- **Reporting**: Detailed visual and numerical reports.
- **History Tracking**: Maintain a record of previous analyses.

## Tech Stack
- **Backend**: Python, Flask, SQLAlchemy
- **Frontend**: HTML5, CSS3 (Modern UI), JavaScript
- **ML/AI**: PyTorch/TensorFlow (Segmentation & Regression)

## Setup and Installation

### Prerequisites
- Python 3.10+
- `pip`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd aireal-data
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the database:
   ```bash
   python init_db.py
   ```
4. Configure environment variables in `.env`:
   ```env
   SECRET_KEY=your_secret_key
   DATABASE_URL=sqlite:///soil_crack.db
   CLOUDINARY_URL=your_cloudinary_url
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## License
MIT License
