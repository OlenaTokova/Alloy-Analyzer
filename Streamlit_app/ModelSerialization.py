from joblib import dump, load

# Function to save a model
def save_model(model, filename):
    dump(model, filename)
    print(f"Model saved successfully as {filename}.")

# Function to load a model
def load_model(filename):
    model = load(filename)
    print(f"Model loaded successfully from {filename}.")
    return model
