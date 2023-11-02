import joblib

# Load your trained model
model = load_trained_model()  # Replace with your model loading logic

# Serialize the model to a file
joblib.dump(model, 'trained_model.pkl')  # Replace with your model serialization logic

# Later, you can deserialize the model
loaded_model = joblib.load('trained_model.pkl')  # Replace with your model deserialization logic
