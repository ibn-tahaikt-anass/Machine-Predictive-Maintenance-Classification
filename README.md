Predictive Maintenance Classification using PyTorch MLP

This repository contains a PyTorch implementation of a Multilayer Perceptron (MLP) for predictive maintenance classification. The model is designed to predict the likelihood of a machine undergoing maintenance based on input features from a CSV dataset.

Dataset Definition
The dataset is loaded and preprocessed using the CSVDataset class. The input CSV file should contain features in columns 3 to 8 and the target labels in the last column. The target labels are label-encoded using the LabelEncoder. The dataset can be split into training and testing sets using the get_splits method.

MLP Architecture
The MLP model is defined in the MLP class, which inherits from PyTorch's Module. The architecture consists of three fully connected hidden layers with ReLU activation functions. The number of neurons in each layer is 10, 8, and 2, respectively. Dropout is applied after the first hidden layer to prevent overfitting. The final layer uses the softmax activation function for multi-class classification.

Training the Model
The training process is implemented in the train_model function. It uses the Adam optimizer with a learning rate of 0.01 and L2 regularization. The training loop runs for a specified number of epochs (default is 100), and early stopping is employed to prevent overfitting. The training loss per epoch is stored for later analysis.

Evaluation
The evaluate_model function calculates the accuracy of the trained model on the test set. Additionally, a predict function is provided to make predictions on new data.
