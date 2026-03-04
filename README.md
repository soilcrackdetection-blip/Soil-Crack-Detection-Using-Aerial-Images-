---
title: Soil Crack Detection
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

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
   DATABASE_URL=postgresql://user:pass@host:port/dbname
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## Model Weights
**Note**: Due to data size constraints, the pre-trained model weights (`.pth` files) are excluded from this repository via `.gitignore`. 

To use the platform:
1.  **Training**: Run the scripts in the `training/` directory (e.g., `python training/classification.py`, `python training/segmentation.py`, etc.) to generate the models.
2.  **Output**: After training, the models will be automatically saved in the `weights/` folder.
3.  **Deployment**: If you are deploying to a server (like Render), you must ensure these weights are uploaded to your server using **Git LFS**.

## License
MIT License
