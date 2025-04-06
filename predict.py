import pickle
import numpy as np

# Load the saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Example input: Sepal Length, Sepal Width, Petal Length, Petal Width
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # You can change these values

# Make a prediction
prediction = model.predict(sample_input)

# Mapping numerical output to species names
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
predicted_species = species_map[prediction[0]]

print(f"Predicted Iris Species: {predicted_species}")
