import numpy as np
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def Activation(z, acitve):
    def none(z):
        return z
    
    def none_derivative(z):
        return np.ones_like(z)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(a):
        return a * (1 - a)

    def relu(z):
        return np.maximum(0, z)

    def relu_derivative(z):
        return (z > 0).astype(float)

    def tangh(z):
        return np.tanh(z)
    
    def tangh_derivative(a):
        return 1 - a ** 2
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def softmax_derivative(a):
        return a * (1 - a)
    
    dict_1 = {
        "sigmoid": sigmoid,
        'softmax': softmax,
        "relu": relu,
        "tangh": tangh,
        "tangh_derivative": tangh_derivative,
        "sigmoid_derivative": sigmoid_derivative,
        "relu_derivative": relu_derivative,
        "softmax_derivative": softmax_derivative,
        "None": none,
        "None_derivative": none_derivative
    }

    return dict_1[acitve](z)

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu', lr=0.05):
        self.activation = activation
        self.lr = lr
        self.X = None
        self.Z = None
        self.A = None
        self.layer_id = id(self)

        if activation == 'relu':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        elif activation in ['sigmoid', 'tangh']:
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = Activation(self.Z, self.activation)
        return self.A

    def backward(self, dA):
        if self.activation == 'sigmoid':
            dZ = dA * Activation(self.A, 'sigmoid_derivative')
        elif self.activation == 'softmax':
            dZ = dA
        else:
            dZ = dA * Activation(self.Z, self.activation + '_derivative')

        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T
        return dX


    def update_params(self, optimizer):
        params = {f"{self.layer_id}_W": self.W, f"{self.layer_id}_b": self.b}
        grads = {f"{self.layer_id}_W": self.dW, f"{self.layer_id}_b": self.db}
        updates = optimizer.update(params, grads)
        self.W -= updates[f"{self.layer_id}_W"]
        self.b -= updates[f"{self.layer_id}_b"]

class ConvLayer:
    def __init__(self, input_depth, num_filters, filter_size, stride=1, padding=0, lr=0.01):
        self.input_depth = input_depth
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.lr = lr
        self.layer_id = id(self)

        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) * 0.01
        self.biases = np.zeros((num_filters, 1))
        self.X = None
        self.dFilters = None
        self.dBiases = None

    def forward(self, X):
        self.X = X
        batch_size, input_depth, input_height, input_width = X.shape

        out_height = (input_height - self.filter_size + 2 * self.padding) // self.stride + 1
        out_width = (input_width - self.filter_size + 2 * self.padding) // self.stride + 1

        out = np.zeros((batch_size, self.num_filters, out_height, out_width))

        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X

        for b in range(batch_size):
            # Vectorized extraction of sliding patches and convolution using tensordot
            # X_padded shape: (batch_size, input_depth, H_p, W_p)
            s0, s1, s2, s3 = X_padded.strides
            # Create a view of sliding windows: shape -> (batch, input_depth, out_h, out_w, k, k)
            patches_shape = (batch_size, input_depth, out_height, out_width, self.filter_size, self.filter_size)
            patches_strides = (s0, s1, s2 * self.stride, s3 * self.stride, s2, s3)
            patches = np.lib.stride_tricks.as_strided(X_padded, shape=patches_shape, strides=patches_strides)

            # Perform convolution via tensor contraction between patches and filters
            # Result from tensordot: (batch, out_h, out_w, num_filters)
            conv = np.tensordot(patches, self.filters, axes=([1, 4, 5], [1, 2, 3]))
            # Rearrange to (batch, num_filters, out_h, out_w) and add biases
            out = conv.transpose(0, 3, 1, 2) + self.biases.reshape(1, self.num_filters, 1, 1)

        return out
    
    def backward(self, dOut):
        batch_size, _, out_height, out_width = dOut.shape
        _, input_depth, input_height, input_width = self.X.shape

        # Pad input if necessary
        if self.padding > 0:
            X_padded = np.pad(self.X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_padded = self.X

        dX_padded = np.zeros_like(X_padded)
        self.dFilters = np.zeros_like(self.filters)
        self.dBiases = np.zeros((self.num_filters, 1))

        for f in range(self.num_filters):
            for i in range(out_height):
                for j in range(out_width):
                    vert_start = i * self.stride
                    vert_end = vert_start + self.filter_size
                    horiz_start = j * self.stride
                    horiz_end = horiz_start + self.filter_size

                    region = X_padded[:, :, vert_start:vert_end, horiz_start:horiz_end]
                    self.dFilters[f] += np.sum(region * dOut[:, f:f+1, i:i+1, j:j+1], axis=0)
                    self.dBiases[f, 0] += np.sum(dOut[:, f, i, j])
                    dX_padded[:, :, vert_start:vert_end, horiz_start:horiz_end] += self.filters[f] * dOut[:, f:f+1, i:i+1, j:j+1]

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        return dX

    def update_params(self, optimizer):
        params = {f"{self.layer_id}_filters": self.filters, f"{self.layer_id}_biases": self.biases}
        grads = {f"{self.layer_id}_filters": self.dFilters, f"{self.layer_id}_biases": self.dBiases}
        updates = optimizer.update(params, grads)
        self.filters -= updates[f"{self.layer_id}_filters"]
        self.biases -= updates[f"{self.layer_id}_biases"]
    

class FlattenLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dA):
        return dA.reshape(self.input_shape)


class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, X):
        if self.training:
            self.mask = np.random.rand(*X.shape) > self.dropout_rate
            return X * self.mask / (1 - self.dropout_rate)
        else:
            return X

    def backward(self, dA):
        if self.training:
            return dA * self.mask / (1 - self.dropout_rate)
        else:
            return dA


class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        updates = {}
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            updates[key] = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updates


class BCELoss:
    @staticmethod
    def forward(y_pred, y_true):
        eps = 1e-8
        return -np.mean(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )

    @staticmethod
    def backward(y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]


class CrossEntropyLoss:
    @staticmethod
    def forward(logits, y_true):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        loss = -np.mean(np.sum(y_true * np.log(probs + 1e-8), axis=1))
        return loss

    @staticmethod
    def backward(logits, y_true):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        return (probs - y_true) / y_true.shape[0]



class NeuralNetwork:
    def __init__(self, optimizer=None, loss='bce'):
        self.layers = []
        self.optimizer = optimizer if optimizer else AdamOptimizer()
        self.loss_fn = BCELoss if loss == 'bce' else CrossEntropyLoss

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, epochs=100, batch_size=32, validation_data=None):
        logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            epoch_start_time = time.time()
            logger.info(f"Epoch {epoch + 1}/{epochs} started")

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                A = X_batch
                for layer in self.layers:
                    if hasattr(layer, 'training'):
                        layer.training = True
                    A = layer.forward(A)

                # Compute loss
                loss = self.loss_fn.forward(A, y_batch)
                epoch_loss += loss * (end_idx - start_idx)

                # Backward pass
                dA = self.loss_fn.backward(A, y_batch)
                for layer in reversed(self.layers):
                    dA = layer.backward(dA)

                # Update parameters
                for layer in self.layers:
                    if hasattr(layer, 'update_params'):
                        layer.update_params(self.optimizer)

                if batch % 10 == 0:
                    logger.debug(f"Batch {batch + 1}/{n_batches}, Loss: {loss:.4f}")

            avg_loss = epoch_loss / n_samples
            epoch_time = time.time() - epoch_start_time

            # Validation
            val_loss = None
            val_accuracy = None
            if validation_data:
                val_X, val_y = validation_data
                val_pred = self.predict(val_X)
                val_loss = self.loss_fn.forward(val_pred, val_y)
                val_accuracy = self.accuracy(val_pred, val_y)

            logger.info(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s, Train Loss: {avg_loss:.4f}" +
                       (f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}" if validation_data else ""))

    def predict(self, X):
        A = X
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
            A = layer.forward(A)
        return A

    def accuracy(self, y_pred, y_true):
        if y_true.ndim == 1:
            predictions = (y_pred > 0.5).astype(int).flatten()
            return np.mean(predictions == y_true)
        else:
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_labels)
