import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

molecules = np.random.rand(100, 1)  # Heat of formation as input feature
temperatures = 300 + 50 * molecules + np.random.normal(0, 10, (100, 1))  # Decomposition temperatures

# Split the data into training and testing sets
molecules_train, molecules_test, temperatures_train, temperatures_test = train_test_split(
    molecules, temperatures, test_size=0.2, random_state=42
)

# Create and train a linear regression model
model = LinearRegression()
model.fit(molecules_train, temperatures_train)

# Make predictions on the test set
temperatures_pred = model.predict(molecules_test)

# Evaluate the model
mse = mean_squared_error(temperatures_test, temperatures_pred)
print(f'Mean Squared Error on Test Set: {mse}')

if __name__ == "__main__":
    heat_of_formation_of_molecule = np.array([[0.8]])

    # Predict the decomposition temperature for the given molecule
    predicted_temperature = model.predict(heat_of_formation_of_molecule)
    print(f'Predicted Decomposition Temperature: {predicted_temperature[0][0]} K')
