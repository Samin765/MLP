# Neural Network Classifier

This repository contains a simple implementation of a Multi-Layer Perceptron (MLP) neural network classifier for binary classification. It generates synthetic datasets and trains a neural network to classify two different classes.

## Features

- Option to generate either **linear** or **non-linear** datasets.
- Implements forward and backward propagation for the MLP.
- Adjustable hyperparameters: learning rate (`eta`), momentum (`alpha`), epochs, and number of hidden nodes.
- Visualizes training progress with loss curves and accuracy.
- Evaluates the classifier's performance using test data.

## Code Structure

- **Data Generation**: Generates two classes of data (`classA` and `classB`) with user-defined means and variances for each class.
- **Training & Testing Split**: Allows splitting the dataset into training and testing sets using the `splitClassesUnequal` function.
- **MLP Implementation**: Defines a two-layer MLP and performs training using backpropagation.
- **Evaluation**: Computes the accuracy of the trained model and displays results.
- **Visualization**: Plots the data and classification boundaries.

## Parameters

You can adjust the following parameters:

- `non_linear_data`: Boolean to toggle between generating linear or non-linear datasets.
- `n`: Number of data points per class.
- `nodes`: Number of nodes in the hidden layer.
- `alpha`: Momentum coefficient.
- `eta`: Learning rate.
- `epochs`: Number of training epochs.
- `ratioA`, `ratioB`: Ratios of training data for class A and class B, respectively.

## Functions

- `generateClasses(n, mA, mB, sigmaA, sigmaB, non_linear_data, ratioA, ratioB)`: Generates the datasets for class A and class B.
- `splitClassesUnequal(classA, classB, numTrainA, numTrainB)`: Splits the classes into training and testing datasets.
- `MLP(X, X_test, y, y_test, W1, W2, nodes, n, epochs, alpha, eta, dW1, dW2, n_test)`: Trains the MLP using forward and backward passes.
- `evluateClassifier(X, y, W1Star, W2Star, classA_test, classB_test, n_test)`: Evaluates the trained model and displays accuracy.
- `plotData(X, y, W1, W2, classA, classB)`: Plots the class data and decision boundary.

## Usage

1. **Install necessary dependencies**:
   Ensure you have a MATLAB environment set up as this implementation is MATLAB-based.

2. **Run the code**:
   Adjust the parameters as needed in the script and run it to generate data, train the model, and evaluate it.

3. **Visualize results**:
   The script will plot the data and display training and validation loss as well as accuracy metrics.

## Example

```matlab
non_linear_data = true;
n = 100;
nodes = 40;
alpha = 0.9;
eta = 0.05;
epochs = 200;
ratioA = 0.5;
ratioB = 1;
[classA, classB] = generateClasses(n, mA, mB, sigmaA, sigmaB, non_linear_data, ratioA, ratioB );
[X, y, W1, W2, dW1, dW2] = generateData(classA_train, classB_train, n, nodes);
[W1Star, W2Star] = MLP(X, X_test, y, y_test, W1, W2, nodes, n, epochs, alpha, eta, dW1, dW2, n_test);
evluateClassifier(X_test, y_test, W1Star, W2Star, classA_test, classB_test, n_test);
