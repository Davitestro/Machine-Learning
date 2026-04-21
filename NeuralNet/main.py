from NN import NeuralNetwork, DenseLayer, ConvLayer, FlattenLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def make_lines(n=1000, img_size=16):
    X = np.zeros((n, 1, img_size, img_size))
    y = np.zeros((n, 2))  # Change to 2 classes for one-hot encoding

    for i in range(n):
        if np.random.rand() > 0.5:
            col = np.random.randint(2, img_size-2)
            X[i, 0, :, col] = 1.0   # вертикальная
            y[i] = [1, 0]  # Vertical: [1, 0]
        else:
            row = np.random.randint(2, img_size-2)
            X[i, 0, row, :] = 1.0   # горизонтальная
            y[i] = [0, 1]  # Horizontal: [0, 1]

    return X, y


# def generate_nn_friendly_dataset(
#     n_samples=1000,
#     n_features=2,
#     noise=0.1,
#     radius=1.0,
#     seed=42
# ):
#     np.random.seed(seed)

#     X = np.random.uniform(-1.5, 1.5, size=(n_samples, n_features))

#     dist = np.linalg.norm(X, axis=1)

#     y = (dist < radius).astype(int)

#     X += np.random.normal(0, noise, size=X.shape)

#     return X, y


def evaluate_model(model_name, model, X_test, y_test):
    probs = model.predict(X_test)

    print("DEBUG shapes:")
    print("probs:", probs.shape)
    print("y_test:", y_test.shape)

    # For multi-class classification
    if y_test.shape[1] > 1:  # One-hot encoded
        preds = np.argmax(probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
    else:  # Binary classification
        preds = (probs > 0.5).astype(int).reshape(-1)
        y_true = y_test.reshape(-1)

    acc = np.mean(preds == y_true)

    print(f"\n--- {model_name} ---")
    print("Accuracy:", acc)
    print("Predictions distribution:")
    print("Class 0 (Vertical):", np.sum(preds == 0))
    print("Class 1 (Horizontal):", np.sum(preds == 1))
    
    return acc


# x, y = generate_nn_friendly_dataset(n_samples=5000, noise=0.1, n_features=6, radius=1.0, seed=42)


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, seed=42)
# y_train = y_train.reshape(-1, 1)


# NerN = NeuralNetwork()
# NerN.add(DenseLayer(6, 16, activation='relu', lr=0.05))
# NerN.add(DenseLayer(16, 16, activation='relu', lr=0.05))
# NerN.add(DenseLayer(16, 1, activation='sigmoid', lr=0.05))

# NerN.fit(x_train, y_train, epochs=200)
# y_pred = NerN.predict(x_test)
# y_pred = y_pred.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)


# print("accuracy:", np.mean((y_pred > 0.5) == y_test))


from NN import NeuralNetwork, DenseLayer, ConvLayer, FlattenLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def make_lines(n=1000, img_size=16):
    X = np.zeros((n, 1, img_size, img_size))
    y = np.zeros((n, 2))  # Change to 2 classes for one-hot encoding

    for i in range(n):
        if np.random.rand() > 0.5:
            col = np.random.randint(2, img_size-2)
            X[i, 0, :, col] = 1.0   # вертикальная
            y[i] = [1, 0]  # Vertical: [1, 0]
        else:
            row = np.random.randint(2, img_size-2)
            X[i, 0, row, :] = 1.0   # горизонтальная
            y[i] = [0, 1]  # Horizontal: [0, 1]

    return X, y


def evaluate_model(model_name, model, X_test, y_test):
    probs = model.predict(X_test)

    print("DEBUG shapes:")
    print("probs:", probs.shape)
    print("y_test:", y_test.shape)

    # For multi-class classification
    if y_test.shape[1] > 1:  # One-hot encoded
        preds = np.argmax(probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
    else:  # Binary classification
        preds = (probs > 0.5).astype(int).reshape(-1)
        y_true = y_test.reshape(-1)

    acc = np.mean(preds == y_true)
    return acc


# Generate data with one-hot encoded labels
x, y = make_lines(n=1000, img_size=16)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create model with cross-entropy loss
model = NeuralNetwork(loss='cross_entropy')

# Build CNN architecture
model.add(ConvLayer(1, 8, 3, padding=1))  # 8 filters, 3x3
model.add(FlattenLayer())
model.add(DenseLayer(8 * 16 * 16, 16, activation='relu'))
model.add(DenseLayer(16, 2, activation="softmax"))  # Softmax for 2 classes

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate
accuracy = evaluate_model("CNN Line Classifier", model, x_test, y_test)

# Test some examples
print("\n--- Testing individual examples ---")
for i in range(5):
    sample = x_test[i:i+1]
    true_label = np.argmax(y_test[i])
    prediction = model.predict(sample)
    pred_class = np.argmax(prediction)
    print(f"Sample {i}: True={true_label}, Predicted={pred_class}, "
          f"Probabilities={prediction[0].round(4)}")