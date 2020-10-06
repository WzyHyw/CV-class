import numpy as np
import sys
sys.path.append('mnist')
import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

n_train, w, h = train_images.shape # each picture 28*28

X_train = train_images.reshape((n_train, w * h)) # flatten,(60000*784)
Y_train = train_labels

n_test, w, h = test_images.shape
X_test = test_images.reshape((n_test, w * h)) # (10000,784)
Y_test = test_labels

N = X_train.shape[0] # 60000
N_split = int(0.9 * N) # 90% for training, 10% for validation

mask = list(range(N_split))
x_train = X_train[mask]
y_train = Y_train[mask]

mask = list(range(N_split, N))
x_val = X_train[mask]
y_val = Y_train[mask]


class ThreeLayerNet(object):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, std=5e-2):
        self.params = {}
        # initialization
        self.params['W1'] = std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        scores = None

        N, D = X.shape

        # forward propagation
        z1 = X.dot(W1) + b1
        h1 = np.maximum(0, z1)
        z2 = h1.dot(W2) + b2
        h2 = np.maximum(0, z2)
        scores = h2.dot(W3) + b3

        if y is None:
            return scores
        loss = None
        C = scores.shape[1]
        sco = np.zeros((N, C))
        sco = scores - np.max(scores, axis=1, keepdims=True) # subtract the biggest score for each
        p = np.exp(sco) / np.sum(np.exp(sco), axis=1, keepdims=True) # softmax

        y_label = np.zeros((N, C))
        y_label[np.arange(N), y] = 1

        loss = (-1) * np.sum(np.multiply(np.log(p), y_label)) / N # cross-entropy loss
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)) # loss for regularization
        grads = {}

        dZ3 = p - y_label
        dW3 = h2.T.dot(dZ3) / N + 2 * reg * W3
        db3 = np.sum(dZ3, axis=0) / N

        dZ2 = (dZ3).dot(W3.T) * (h2 > 0)
        dW2 = h1.T.dot(dZ2) / N + 2 * reg * W2
        db2 = np.sum(dZ2, axis=0) / N

        dZ1 = (dZ2).dot(W2.T) * (h1 > 0)
        dW1 = X.T.dot(dZ1) / N + 2 * reg * W1
        db1 = np.sum(dZ1, axis=0) / N

        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1
        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        print(iterations_per_epoch)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_idx = np.random.choice(num_train, batch_size)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['W3'] -= learning_rate * grads['W3']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']
            self.params['b3'] -= learning_rate * grads['b3']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay # decay the learning rate after each epoch

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        y_pred = None
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)

        return y_pred


input_size = 784
hidden_size1 = 128
hidden_size2 = 64
num_classes = 10
net = ThreeLayerNet(input_size, hidden_size1, hidden_size2, num_classes)

stats = net.train(x_train, y_train, x_val, y_val,
                  num_iters=5000, batch_size=200,
                  learning_rate=7e-3, learning_rate_decay=0.95,
                  reg=0.25, verbose=True)

val_acc = (net.predict(x_val) == y_val).mean()
print('Validation accuracy: ', val_acc)
val_acc = (net.predict(X_test) == Y_test).mean()
print('Test accuracy: ', val_acc)
