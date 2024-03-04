import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, max_epochs=1000):
        self.weights = np.random.rand(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # adding bias
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.max_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def test(self, test_inputs, labels):
        predictions = [self.predict(inputs) for inputs in test_inputs]
        accuracy = np.mean(predictions == labels)
        return accuracy


def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    inputs = data[:, :-1]
    labels = data[:, -1]
    return inputs, labels


def plot_data(inputs, labels, weights):
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap=plt.cm.coolwarm)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    
    x_vals = np.array([-2, 2])
    y_vals = -(weights[1] * x_vals + weights[0]) / weights[2]
    plt.plot(x_vals, y_vals, '--', color='black')
    
    plt.show()


if __name__ == "__main__":

    training_inputs, training_labels = read_data('C:/Users/evert/Desktop/2024-A/Seminario IA2/Practica 1/Nueva carpeta/OR_trn.csv')
    test_inputs, test_labels = read_data('C:/Users/evert/Desktop/2024-A/Seminario IA2/Practica 1/Nueva carpeta/OR_tst.csv')

    perceptron = Perceptron(input_size=2)
    perceptron.train(training_inputs, training_labels)

    accuracy = perceptron.test(test_inputs, test_labels)
    print(f"Accuracy on test data: {accuracy}")

    plot_data(training_inputs, training_labels, perceptron.weights)
