import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21 ,1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)

def loss(y, y_hat):
    # Mean Square Error
    return np.mean((y-y_hat)**2)

def derivative_loss(y, y_hat):
    return (y-y_hat)*(2/y.shape[0])

class layer():
    def __init__(self, input_size, output_size):
        self.weight = np.random.normal(0, 1, (input_size+1, output_size))

    def forward(self, x):
        # Append bias into the x
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.forward_gradient = x
        self.y = sigmoid(np.matmul(x, self.weight))
        return self.y

    def backward(self, grad):
        #\frac{\partial C}{\partial \omega^{(L)}} = \frac{\partial z^{(L)}}{\partial \omega^{(L)}} \frac{\partial a^{(L)}}{\partial z^{(L)}} \frac{\partial C_0}{\partial a^{(L)}}
        self.backward_gradient = np.multiply(derivative_sigmoid(self.y), grad)
        return np.matmul(self.backward_gradient, self.weight[:-1].T)

    def update(self, learning_rate):
        self.gradient = np.matmul(self.forward_gradient.T, self.backward_gradient)
        self.weight -= learning_rate * self.gradient
        return self.gradient

class Dense():
    def __init__(self, width, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.layer = [layer(before, after) for before, after in zip(width[:-1], width[1:])]

    def forward(self, x):
        result = x
        for lay in self.layer:
            result = lay.forward(result)
        return result

    def backward(self, grad):
        result = grad
        for lay in self.layer[::-1]:
            result = lay.backward(result)
        return result

    def update(self):
        gradients = [lay.update(self.learning_rate) for lay in self.layer]
        return gradients

if __name__ == '__main__':
    nn_linear = Dense([2, 4, 4, 1], 1)
    nn_xor = Dense([2, 4, 4, 1], 1)

    epoch = 100000
    loss_tol = 0.005

    print('{}\nLinear Model has started\n{}'.format('-'*100, '-'*100))
    x_linear, y_linear = generate_linear()
    count = 0
    for i in range(epoch):
        y = nn_linear.forward(x_linear)
        loss_linear = loss(y, y_linear)
        nn_linear.backward(derivative_loss(y, y_linear))
        nn_linear.update()

        if (i+1)%5000 == 0:
            print('Epoch {} loss: {}'.format(i+1, loss_linear))

        if loss_linear < loss_tol:
            count += 1
            if count == 5:
                print('Linear Model Performance is satisfactory')
                break
        else:
            count = 0

    print('{}\nXOR Model has started\n{}'.format('-'*100, '-'*100))
    x_xor, y_xor = generate_XOR_easy()
    count = 0
    for i in range(epoch):
        y = nn_xor.forward(x_xor)
        loss_xor = loss(y, y_xor)
        nn_xor.backward(derivative_loss(y, y_xor))
        nn_xor.update()

        if (i+1)%5000 == 0:
            print('Epoch {} loss: {}'.format(i+1, loss_xor))

        if loss_xor < loss_tol:
            count += 1
            if count == 5:
                print('XOR Model Performance is satisfactory')
                break
        else:
            count = 0

    y1 = nn_linear.forward(x_linear)
    show_result(x_linear, y_linear, y1)
    print('linear test loss : ', loss(y1, y_linear))
    print('linear test accuracy : {:3.2f}%'.format(np.count_nonzero(np.round(y1) == y_linear) * 100 / len(y1)))
    y2 = nn_xor.forward(x_xor)
    show_result(x_xor, y_xor, y2)
    print('XOR test loss : ', loss(y2, y_xor))
    print('XOR test accuracy : {:3.2f}%'.format(np.count_nonzero(np.round(y2) == y_xor) * 100 / len(y2)))
    print('\n linear test result : \n',y1)
    print('\n XOR test result : \n',y2)