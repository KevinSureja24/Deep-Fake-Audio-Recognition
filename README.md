# Deep-Fake-Audio-Recognition
Overview
The Deep Fake Audio Recognition project aims to develop a system capable of detecting deep fake audio recordings using Convolutional Neural Networks (CNNs). The system utilizes image data generated from audio spectrograms to train and evaluate the CNN model. The model is trained on a dataset containing both real and deep fake audio images to distinguish between authentic and manipulated audio recordings.

Features
CNN Architecture: The system employs a CNN architecture consisting of convolutional layers, pooling layers, dropout layers, and fully connected layers to learn discriminative features from audio spectrogram images.
Data Augmentation: Image data augmentation techniques such as shearing, zooming, and horizontal flipping are applied to the training dataset to improve model generalization and robustness.
Evaluation Metrics: The system evaluates the trained model using accuracy metrics, including training accuracy, validation accuracy, and test accuracy, to assess its performance in distinguishing between real and deep fake audio images.
Prediction: After training, the model can predict whether an audio image is real or a deep fake with associated confidence scores.
Usage
To use the Deep Fake Audio Recognition system:

Data Preparation: Organize your audio data into separate directories for training, validation, and testing.
Model Training: Execute the provided code to train the CNN model on the training dataset. Adjust parameters such as epochs, batch size, and network architecture as needed.
Model Evaluation: Evaluate the trained model using the validation dataset to monitor its performance and adjust hyperparameters accordingly.
Testing: Test the final model on unseen data from the testing dataset to assess its real-world performance and obtain accuracy metrics.
Prediction: Use the trained model to predict whether an audio image is real or a deep fake by providing the image path.
Requirements
Python 3.x
Keras
TensorFlow
Matplotlib
OpenCV
NumPy
scikit-learn
Ensure that all dependencies are installed before running the code.

Contributors
Dev Dubal

Acknowledgments
We would like to acknowledge the contributions of the open-source community and the developers of libraries and frameworks used in this project.

Feel free to contribute, report issues, or suggest improvements to the Deep Fake Audio Recognition project. Your feedback is valuable in enhancing the accuracy and effectiveness of the system.
