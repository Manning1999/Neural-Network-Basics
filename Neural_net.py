# This is an extremely simple neural network with no hidden layers that will take a sequence of three numbers.
# If the first number is a 0 then the output will be 0. If the first number of 1 then the output should be 1
# This project is based of a tutorial that can be found at https://www.youtube.com/watch?v=kft1AJ9WVDk but I
# refactored it to further compartmentalize it more as well as save the neural network and give the option of using
# the saved network or retraining it

import pickle
import numpy as np





class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iterations in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


def load_from_file():
    neural_network.synaptic_weights = pickle.load(open('weights.pkl', 'rb'))
    print("Loaded saved weights")
    get_input()


def retrain():
    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1],
                                [1, 1, 1],
                                [1, 2, 1],
                                [1, 8, 1],
                                [1, 7, 1],
                                [1, 3, 1],
                                [1, 4, 1],
                                [0, 2, 1],
                                [1, 4, 1],
                                [1, 1, 1],
                                [0, 4, 0],
                                [0, 3, 0],
                                [0, 9, 1],
                                [0, 6, 1],
                                [0, 9, 0],
                                [0, 1, 1]])

    training_outputs = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]]).T

    neural_network.train(training_inputs, training_outputs, 100000)

    print("synaptic weights after training")
    print(neural_network.synaptic_weights)
    pickle.dump(neural_network.synaptic_weights, open('weights.pkl', 'wb'))
    get_input()


def ask():
    isloadingPreviousWeights = str(input("Load weights from file? T/F    ")).capitalize()


    if isloadingPreviousWeights == 'T':
        # attempt to load previously saved weights. If there is none then train the network from scratch
        try:
            load_from_file()

        except:
            print("Couldn't find weights.pkl file and will retrain the network")
            retrain()

    elif isloadingPreviousWeights == 'F':
        retrain()

    else:
        print('Invalid Input')
        ask()


def get_input():
    a = str(input("Input1: "))
    b = str(input("Input2: "))
    c = str(input("Input3: "))

    print("New Situation: Input Data == ", a,b,c)
    print("Output data: ")
    print(neural_network.think(np.array([a,b,c])))


if __name__ == "__main__":

    # Create new neural network
    neural_network = NeuralNetwork()

    ask()



