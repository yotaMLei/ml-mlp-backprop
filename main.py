"""
Introduction to Machine Learning - Programming Assignment
Exercise 06
January 2021
Yotam Leibovitz
"""
import matplotlib.pyplot as plt
import numpy as np
import time


# runtime measurement
def tic():
    """
    runtime measurement
    """
    return time.time()


def toc(t):
    """
    runtime measurement
    """
    return float(tic()) - float(t)


def sigma(a):
    """
    Logistic sigmoid function
    a = input vector
    """
    return 1 / (1 + np.exp(-a))


class FCNeuralNet:
    """
    Implementation of the given fully connected neural network (FC NN) with 3 or 6 hidden units
    100 Instances in parallel
    """
    def __init__(self, six_hidden_units=False):
        self.six_hidden_units = six_hidden_units  # True or False, default is 3 hidden units
        if self.six_hidden_units:
            self.Z = np.zeros((100, 6, 1))  # vector of hidden units
            self.y = np.zeros((100, 1, 1))   # scalar output
            self.W_1 = np.zeros((100, 6, 4))  # weight matrix of the 1st layer
            self.W_2 = np.zeros((100, 1, 7))  # weight vector of the 2nd layer
            self.A_1 = np.zeros((100, 6, 1))  # a vector of the 1st layer (sigma(a) = z)
            self.A_2 = np.zeros((100, 1, 1))  # a scalar of the 2nd layer (sigma(a) = z)
            self.D_1 = np.zeros((100, 6, 1))  # delta vector of the 1st layer
            self.D_2 = np.zeros((100, 1, 1))  # delta scalar of the output unit
        else:  # the default is 3 hidden units
            self.Z = np.zeros((100, 3, 1))  # vector of hidden units
            self.y = np.zeros((100, 1, 1))  # scalar output
            self.W_1 = np.zeros((100, 3, 4))  # weight matrix of the 1st layer
            self.W_2 = np.zeros((100, 1, 4))  # weight vector of the 2nd layer
            self.A_1 = np.zeros((100, 3, 1))  # a vector of the 1st layer (sigma(a) = z)
            self.A_2 = np.zeros((100, 1, 1))  # a scalar of the 2nd layer (sigma(a) = z)
            self.D_1 = np.zeros((100, 3, 1))  # delta vector of the 1st layer
            self.D_2 = np.zeros((100, 1, 1))  # delta scalar of the output unit

    def activation(self, a):
        """
        The activation function of the NN
        a = input vector
        """
        return sigma(a)

    def activation_tag(self, a):
        """
        The derivative of the activation function
        a = input vector
        """
        return sigma(a) * (1 - sigma(a))

    def forward_pass(self, x):
        """
        Computes the forward pass
        (on 100 instances in parallel)
        """
        x_bias = np.append(x, 1)

        # reshape x for parallel processing
        x_bias = x_bias.reshape(1, 4, 1)
        x_bias = np.repeat(x_bias, 100, axis=0)

        self.A_1 = self.W_1 @ x_bias

        # reshape to valid vector
        if self.six_hidden_units:
            self.A_1 = self.A_1.reshape((100, 6, 1))
        else:
            self.A_1 = self.A_1.reshape((100, 3, 1))
        self.Z = self.activation(self.A_1)

        # add bias 1 at the end of each z vector
        Z_bias = np.concatenate((self.Z, np.ones((100, 1, 1))), axis=1)

        self.A_2 = self.W_2 @ Z_bias
        self.A_2 = self.A_2.reshape((100, 1, 1))
        self.y = self.activation(self.A_2)

    def backward_pass(self, t):
        """
        Computes the backward pass
        (on 100 instances in parallel)
        """
        self.D_2 = self.activation_tag(self.A_2) * (self.y - t)
        self.D_1 = self.activation_tag(self.A_1) * (np.transpose(self.W_2[:, :, 0:-1], axes=(0, 2, 1)) * self.D_2)

    def back_propagation(self, x, t):
        """
        Computes back propagation algorithm and returns the output y
        (on 100 instances in parallel)
        """
        grad_1 = np.zeros(self.W_1.shape)
        grad_2 = np.zeros(self.W_2.shape)

        self.forward_pass(x)
        self.backward_pass(t)

        x_bias = np.append(x, 1)

        # reshape x for parallel processing
        x_bias = x_bias.reshape(1, 1, 4)
        x_bias = np.repeat(x_bias, 100, axis=0)

        # add bias 1 at the end of each z vector
        Z_bias = np.concatenate((self.Z, np.ones((100, 1, 1))), axis=1)

        if self.six_hidden_units:
            for i in range(100):
                grad_1[i, :, :] = np.outer(self.D_1[i, :, :].reshape(6, 1), x_bias[i, :, :].reshape(4, 1)) # gradient of the weights of the 1st layer
                grad_2[i, :, :] = np.outer(self.D_2[i, :, :].reshape(1,1), Z_bias[i, :, :].reshape(7, 1)) # gradient of the weights of the 1st layer

        else:
            for i in range(100):
                grad_1[i, :, :] = np.outer(self.D_1[i, :, :].reshape(3, 1), x_bias[i, :, :].reshape(4, 1)) # gradient of the weights of the 1st layer
                grad_2[i, :, :] = np.outer(self.D_2[i, :, :].reshape(1,1), Z_bias[i, :, :].reshape(4, 1)) # gradient of the weights of the 1st layer

        return self.y.reshape(100, 1), grad_1, grad_2

    def initialize_random_weights(self):
        """
        Initialize the weights of the NN with random weights taken from normal(0,1) distribution
        (on 100 instances in parallel)
        """
        if self.six_hidden_units:
            self.W_1 = np.random.normal(loc=0, scale=1.0, size=(100, 6, 4))
            self.W_2 = np.random.normal(loc=0, scale=1.0, size=(100, 1, 7))
        else:
            self.W_1 = np.random.normal(loc=0, scale=1.0, size=(100, 3, 4))
            self.W_2 = np.random.normal(loc=0, scale=1.0, size=(100, 1, 4))


class GradientDescentOptimizer:
    """
    Implementation of the batch gradient descent optimization process
    """
    def __init__(self, model, x_batch, t_vec, learning_rate):
        self.model = model # FCNeuralNet object
        self.x_batch = x_batch
        self.t_vec = t_vec
        self.learning_rate = learning_rate
        self.loss = np.zeros((100, 1, 1))
        self.total_grad_1 = np.zeros(self.model.W_1.shape)
        self.grad_1 = np.zeros(self.model.W_1.shape)
        self.total_grad_2 = np.zeros(self.model.W_2.shape)
        self.grad_2 = np.zeros(self.model.W_2.shape)
        self.y_tensor = np.zeros((100, 1, 8))

    def update_weights(self):
        """
        Updates the weights according to batch gradient descent
        """
        self.model.W_1 = self.model.W_1 - self.learning_rate * self.total_grad_1
        self.model.W_2 = self.model.W_2 - self.learning_rate * self.total_grad_2

    def batch_gradient_descent(self):
        """
        Performs batch gradient descent algorithm and computes the loss function
        """
        # obtain the total_gradient
        for n in range(len(self.x_batch)):
            self.y_tensor[:, :, n], self.grad_1, self.grad_2 = self.model.back_propagation(self.x_batch[n], self.t_vec[n])
            self.total_grad_1 += self.grad_1
            self.total_grad_2 += self.grad_2

        # according to the formula of E, gradE should be multiplied by 1/4
        self.total_grad_1 = self.total_grad_1 * (1/4)
        self.total_grad_2 = self.total_grad_2 * (1/4)

        # update weights
        self.update_weights()

        # compute loss function
        for i in range(100):
            self.loss[i, :, :] = (1/8) * np.sum((self.y_tensor[i, :, :] - self.t_vec) ** 2)


if __name__ == "__main__":
    """
    Main function                                    
    """
    NUM_OF_ITERATIONS = 2000
    NUM_OF_TEST_RUNS = 100 # don't change this

    # batch of the 8 possible inputs
    x_batch = [np.array((0, 0, 0)), np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, 0, 1)),
        np.array((1, 1, 0)), np.array((0, 1, 1)), np.array((1, 0, 1)), np.array((1, 1, 1))]

    # parity-3 target (1 odd, 0 even)
    t_vec = np.array((0, 1, 1, 1, 0, 0, 0, 1))

    learning_rate = 2
    mean_losses = [np.zeros(NUM_OF_ITERATIONS), np.zeros(NUM_OF_ITERATIONS)]
    models = [FCNeuralNet(), FCNeuralNet(six_hidden_units=True)]

    """
    ######################################################################################
    #     Section A - FC NN with 3 hidden units, batch gradient descent optimization     #
    ###################################################################################### 
    """
    tt1 = tic() # runtime measurement
    model = models[0]
    mean_loss = mean_losses[0]
    gd_optimizer = GradientDescentOptimizer(model, x_batch, t_vec, learning_rate)

    model.initialize_random_weights()
    for i in range(NUM_OF_ITERATIONS):
        gd_optimizer.batch_gradient_descent()
        mean_loss[i] = np.sum(gd_optimizer.loss) / NUM_OF_TEST_RUNS
        # print("iteration = {0}, final mean loss: {1}".format(i, mean_loss[i]))

    # plot the mean loss w.r.t the iteration i
    i = np.arange(1, NUM_OF_ITERATIONS+1, 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set(title='Section A\nMean loss per iteration', xlabel='Iteration', ylabel='Mean loss over 100 runs', xlim=(1, i[-1]))
    ax1.plot(i, mean_losses[0], linewidth=1.5, c='C3')
    plt.legend(['FC NN With 3 hidden units'])

    print('*** Section A runtime = {:5.4f} sec ***'.format(toc(tt1)))
    fig.tight_layout()
    plt.savefig('ML_ex6_A.png')
    plt.show()

    """
    ######################################################################################
    #     Section B - FC NN with 6 hidden units, batch gradient descent optimization     #
    ###################################################################################### 
    """
    tt2 = tic()  # runtime measurement
    model = models[1]
    mean_loss = mean_losses[1]
    gd_optimizer = GradientDescentOptimizer(model, x_batch, t_vec, learning_rate)

    model.initialize_random_weights()
    for i in range(NUM_OF_ITERATIONS):
        gd_optimizer.batch_gradient_descent()
        mean_loss[i] = np.sum(gd_optimizer.loss) / NUM_OF_TEST_RUNS
        # print("iteration = {0}, final mean loss: {1}".format(i, mean_loss[i]))

    # plot the mean loss w.r.t the iteration i
    i = np.arange(1, NUM_OF_ITERATIONS+1, 1)
    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.set(title='Section B\nMean loss per iteration', xlabel='Iteration', ylabel='Mean loss over 100 runs', xlim=(1, i[-1]))
    ax2.plot(i, mean_losses[1], linewidth=1.5, c='C4')
    plt.legend(['FC NN with 6 hidden units'])

    print('*** Section B runtime = {:5.4f} sec ***'.format(toc(tt2)))
    fig.tight_layout()
    plt.savefig('ML_ex6_B.png')
    plt.show()

    """
    ######################################################################################
    #                  Comparing the results of sections A and B                         #
    ###################################################################################### 
    """
    # plot the mean loss w.r.t the iteration i
    i = np.arange(1, NUM_OF_ITERATIONS + 1, 1)
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.set(title='Comparison of sections A and B\nMean loss per iteration', xlabel='Iteration', ylabel='Mean loss over 100 runs', xlim=(1, i[-1]))
    ax3.plot(i, mean_losses[0], linewidth=1.5, c='C3')
    ax3.plot(i, mean_losses[1], linewidth=1.5, c='C4')
    plt.legend(['FC NN With 3 hidden units', 'FC NN with 6 hidden units'])

    fig.tight_layout()
    plt.savefig('ML_ex6_AB.png')
    plt.show()


