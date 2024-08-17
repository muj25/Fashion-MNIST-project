# Fashion-MNIST-project

# Fashion MNIST Classification with PyTorch
This project demonstrates how to build a deep learning model using PyTorch to classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images in 10 categories, with 60,000 images for training and 10,000 for testing. The images are 28x28 pixels, but in this project, they are resized to 32x32.

## Project Overview
### 1. Data Preprocessing:
Transformations: Applied random rotations, horizontal flips, resizing, and normalization to augment and preprocess the images.
Data Loaders: Created DataLoader objects for both the training and testing datasets to enable efficient batch processing.
### 2. Model Architecture:
Convolutional Layers: Used two convolutional layers followed by max-pooling operations to extract features from the images.
Fully Connected Layers: Flattened the feature maps and passed them through fully connected layers to output class probabilities.
Dropout: Implemented dropout for regularization to prevent overfitting.
### 3. Training the Model:
Loss Function: Used CrossEntropyLoss to calculate the difference between the predicted and actual labels.
Optimizer: Adam optimizer was utilized to update the model's weights.
Training Loop: Trained the model for 10 epochs, recording the training loss and accuracy for each epoch.
### 4. Model Evaluation:
Confusion Matrix: Visualized the performance of the model across different classes.
Classification Report: Generated a report with precision, recall, and F1 scores for each class.
F1 Score: Computed the weighted F1 score to summarize the model's performance.
### 5. Visualization:
Class Distribution: Displayed the distribution of images across different classes in the training set.
Sample Images: Showcased random images from the dataset along with their true and predicted labels.
Learning Curves: Plotted the training loss and accuracy over the epochs to monitor the model's learning progress.
### 6. Saving the Model:
Saved the trained model's state dictionary to a file (fashion_mnist_model.pth) for future use.

## How to Run
### Install Dependencies:
Ensure you have PyTorch, torchvision, NumPy, Matplotlib, Seaborn, and scikit-learn installed.

### Execute the Code:
Run the provided Python script to train the model, visualize results, and save the trained model.

### Model Prediction:
Load the saved model and make predictions on new images using the provided functions.

## Results
The model achieved a weighted F1 score of 0.91, indicating good performance across all classes in the Fashion MNIST dataset.



# Fashion MNIST Classification with TensorFlow
This project demonstrates how to build and train a Convolutional Neural Network (CNN) using TensorFlow to classify images from the Fashion MNIST dataset. The dataset contains 70,000 grayscale images of 10 different fashion categories, with 60,000 images for training and 10,000 for testing.

## Project Overview
### 1. Data Preprocessing:
Normalization: Scaled the pixel values of images to a range of 0 to 1.
Reshaping: Reshaped the images to include a channel dimension, necessary for CNN input.
One-Hot Encoding: Converted the labels to one-hot encoded vectors for multi-class classification.
### 2. Data Augmentation:
Augmentation Techniques: Applied random rotations, shifts, shear, zoom, and horizontal flips to the training images using ImageDataGenerator to enhance model generalization.
### 3. Model Architecture:
Convolutional Layers: Implemented two convolutional layers followed by max-pooling to extract and downsample features.
Fully Connected Layers: Flattened the feature maps and passed them through dense layers for classification.
Dropout: Added dropout to the fully connected layer to prevent overfitting.
### 4. Training the Model:
Compilation: Used the Adam optimizer and categorical cross-entropy loss to compile the model.
Training: Trained the model for 10 epochs with data augmentation, tracking training and validation accuracy and loss.
### 5. Model Evaluation:
Accuracy: Achieved a test accuracy of [insert test accuracy here]%.
Confusion Matrix: Visualized the performance across different classes with a confusion matrix.
Classification Report: Generated a classification report with precision, recall, and F1 scores for each class.
### 6. Visualization:
Training Curves: Plotted the training and validation accuracy and loss over epochs to monitor the model's performance.
Predicted vs. True Labels: Visualized random test images along with the predicted and true labels to verify model predictions.
### 7. Saving the Model:
Saved the trained model to a file (fashion_mnist_cnn.h5) for future inference and reuse.

## How to Run
### Install Dependencies:
Ensure you have TensorFlow, NumPy, Matplotlib, Seaborn, and scikit-learn installed.

### Execute the Code:
Run the provided Python script to train the model, visualize results, and save the trained model.

### Model Prediction:
Load the saved model and make predictions on new or existing images using the provided functions.

Results
The model achieved a test accuracy of 88.28%, indicating good performance in classifying images from the Fashion MNIST dataset.
