import numpy as np


# Define a class for the perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # Initialize weights randomly and set the learning rate
        self.weights = 2 * np.random.random((input_size, 1)) - 1
        self.learning_rate = learning_rate

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of the sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward pass
    def forward(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

    # Training function
    def train(self, training_inputs, training_outputs, iterations):
        for _ in range(iterations):
            # Forward pass
            output = self.forward(training_inputs)

            # Calculate the error
            error = training_outputs - output

            # Calculate adjustments
            adjustments = self.learning_rate * error * self.sigmoid_derivative(output)

            # Update the weights
            self.weights += np.dot(training_inputs.T, adjustments)

    # Predict function to test the trained model
    def predict(self, inputs):
        return self.forward(inputs)


# Input dataset
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

# Output dataset - modified to output 1 when the first input is 0
training_outputs = np.array([[1, 0, 0, 1]]).T

# Initialize the perceptron with 3 input features
perceptron = Perceptron(input_size=3)

# Train the perceptron
perceptron.train(training_inputs, training_outputs, iterations=20000)

# Test the trained model with the training data
print("Output after training:")
output_after_training = perceptron.predict(training_inputs)
print(output_after_training)
