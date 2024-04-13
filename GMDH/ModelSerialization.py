from joblib import dump, load

# Assuming you need to import the GMDH class for some functionality,
# and assuming 'MultilayerGMDH' is the class you intend to use from gmdhpy
from gmdhpy import gmdh

# Function to save a model
def save_model(model, filename):
    dump(model, filename)
    print(f"Model saved successfully as {filename}.")

# Function to load a model
def load_model(filename):
    model = load(filename)
    print(f"Model loaded successfully from {filename}.")
    return model

# If you need to use something from gmdhpy specifically in this file,
# ensure it exists in the package and is correctly referenced.
