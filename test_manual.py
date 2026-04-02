"""
Manual testing script for AutoValor API.
Tests model training, saving, and API functionality.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train_model import VehicleValuationModel


def test_1_create_sample_data():
    """Step 1: Create sample vehicle data."""
    print("\n" + "="*60)
    print("STEP 1: Creating sample training data")
    print("="*60)
    
    # Create sample data with 10 vehicles
    data = {
        'year': [2020, 2019, 2021, 2018, 2022, 2017, 2020, 2019, 2021, 2018],
        'mileage': [50000, 75000, 30000, 120000, 15000, 150000, 55000, 85000, 25000, 110000],
        'cylinders': [4, 6, 4, 8, 4, 6, 4, 6, 4, 8],
        'horsepower': [150, 200, 180, 300, 210, 150, 160, 210, 190, 280],
        'price': [15000, 20000, 25000, 8000, 35000, 5000, 14000, 18000, 26000, 9000]
    }
    
    df = pd.DataFrame(data)
    
    # Create data folder if not exists
    os.makedirs('data', exist_ok=True)
    
    df.to_csv('data/sample_vehicles.csv', index=False)
    print(f"✓ Sample data created: data/sample_vehicles.csv")
    print(f"✓ Records: {len(df)}")
    print("\nSample data:")
    print(df.to_string())
    return df


def test_2_train_model(df):
    """Step 2: Train the model."""
    print("\n" + "="*60)
    print("STEP 2: Training the ML model")
    print("="*60)
    
    # Prepare data
    X = df[['year', 'mileage', 'cylinders', 'horsepower']]
    y = df['price']
    
    # Train model
    model = VehicleValuationModel(model_path='models/vehicle_model.pkl')
    model.train(X, y)
    print("✓ Model training completed")
    
    # Make test predictions
    X_test = X[:3]
    X_scaled = model.scaler.transform(X_test)
    predictions = model.model.predict(X_scaled)
    
    print("\nTest predictions on training data:")
    for idx, (pred, actual) in enumerate(zip(predictions, y[:3])):
        print(f"  Vehicle {idx+1}: Predicted=${pred:,.0f}, Actual=${actual:,.0f}")
    
    return model


def test_3_save_model(model):
    """Step 3: Save the model."""
    print("\n" + "="*60)
    print("STEP 3: Saving model to disk")
    print("="*60)
    
    model.save_model()
    print("✓ Model saved successfully")
    
    # Verify files exist
    if os.path.exists('models/vehicle_model.pkl'):
        size_mb = os.path.getsize('models/vehicle_model.pkl') / (1024*1024)
        print(f"✓ File: models/vehicle_model.pkl ({size_mb:.3f} MB)")
    
    if os.path.exists('models/vehicle_model_scaler.pkl'):
        size_kb = os.path.getsize('models/vehicle_model_scaler.pkl') / 1024
        print(f"✓ File: models/vehicle_model_scaler.pkl ({size_kb:.1f} KB)")


def test_4_load_model():
    """Step 4: Load the model."""
    print("\n" + "="*60)
    print("STEP 4: Loading model from disk")
    print("="*60)
    
    model = VehicleValuationModel(model_path='models/vehicle_model.pkl')
    model.load_model()
    print("✓ Model loaded successfully")
    
    return model


def test_5_make_prediction(model):
    """Step 5: Make predictions with loaded model."""
    print("\n" + "="*60)
    print("STEP 5: Making predictions with loaded model")
    print("="*60)
    
    # Test prediction: 2022 Mustang, 10k miles, 8 cylinders, 450 hp
    features = np.array([[2022, 10000, 8, 450]])
    features_scaled = model.scaler.transform(features)
    prediction = model.model.predict(features_scaled)[0]
    
    print("Test vehicle:")
    print(f"  Year: 2022")
    print(f"  Mileage: 10,000 miles")
    print(f"  Cylinders: 8")
    print(f"  Horsepower: 450")
    print(f"\n✓ Predicted Price: ${prediction:,.2f}")
    
    # Another test: Budget car
    features2 = np.array([[2015, 100000, 4, 120]])
    features2_scaled = model.scaler.transform(features2)
    prediction2 = model.model.predict(features2_scaled)[0]
    
    print("\nTest vehicle 2:")
    print(f"  Year: 2015")
    print(f"  Mileage: 100,000 miles")
    print(f"  Cylinders: 4")
    print(f"  Horsepower: 120")
    print(f"\n✓ Predicted Price: ${prediction2:,.2f}")


def test_6_api_test():
    """Step 6: Test the Flask API."""
    print("\n" + "="*60)
    print("STEP 6: Testing Flask API (manual)")
    print("="*60)
    
    print("\n⚠️  To test the API, run in separate terminal:")
    print("   cd d:\\projects\\autovalor")
    print("   python api/app.py")
    print("\nThen test endpoints:")
    print("\n1. Health check:")
    print("   curl http://localhost:5000/health")
    print("\n2. API Info:")
    print("   curl http://localhost:5000/info")
    print("\n3. Make prediction:")
    print('   curl -X POST http://localhost:5000/predict \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"features": [2022, 10000, 8, 450]}\'')


def main():
    """Run all tests."""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*15 + "AUTOVALOR MANUAL TEST SUITE" + " "*16 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # Run tests sequentially
        df = test_1_create_sample_data()
        model = test_2_train_model(df)
        test_3_save_model(model)
        loaded_model = test_4_load_model()
        test_5_make_prediction(loaded_model)
        test_6_api_test()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📊 Summary:")
        print("  ✓ Sample data created")
        print("  ✓ Model trained")
        print("  ✓ Model saved to disk")
        print("  ✓ Model loaded from disk")
        print("  ✓ Predictions working")
        print("\n👉 Next steps:")
        print("  1. Run: python api/app.py")
        print("  2. Test endpoints with curl")
        print("  3. Run: pytest")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
