# AutoValor - Vehicle Valuation ML API

A Flask-based REST API for vehicle valuation predictions using machine learning.

## Project Structure

```
autovalor/
├── src/                      # Model training code
│   ├── train_model.py       # Model training and persistence
│   └── __init__.py
├── api/                      # Flask application
│   ├── app.py               # Main Flask app and endpoints
│   └── __init__.py
├── tests/                    # Test suite
│   ├── test_model.py        # Model tests
│   ├── test_api.py          # API tests
│   └── __init__.py
├── models/                   # Trained model files (.pkl)
├── data/                     # Training data and datasets
├── requirements.txt          # Python dependencies
├── render.yaml              # Render.com deployment config
├── .gitignore               # Git ignore rules
└── README.md               # This file
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```python
from src.train_model import VehicleValuationModel
import pandas as pd

# Load data
df = pd.read_csv("data/vehicles.csv")
X = df.drop("price", axis=1)
y = df["price"]

# Train and save
model = VehicleValuationModel()
model.train(X, y)
model.save_model()
```

### Running the API

```bash
python api/app.py
```

The API will start on `http://localhost:5000`

### API Endpoints

- **GET `/health`** - Health check
- **GET `/info`** - API information
- **POST `/predict`** - Make a valuation prediction

#### Example Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [2020, 50000, 4, 2.0, 150]}'
```

## Testing

```bash
pytest
pytest --cov=src  # With coverage
```

## Deployment

This project is configured for deployment on [Render.com](https://render.com/) using `render.yaml`.

## License

MIT
