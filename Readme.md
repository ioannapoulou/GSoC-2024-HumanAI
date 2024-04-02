# RenAIssance Project Test based on CRNN Architecture

## Abstract

This project is a test for the RenAIssance project. The goal of this assignment is to create a model based on the CRNN architecture that can recognize text in images of Spanish books from the 17th century. CRNN is the acronym for Convolutional Recurrent Neural Network, which is a type of neural network that combines convolutional layers with recurrent layers to process sequences of data.

## Dataset

For the purpose of this project, two datasets were used. The first dataset is the one provided by the RenAIssance project, which contains images of Spanish books from the 17th century. The second [dataset](https://github.com/PedroBarcha/old-books-dataset) is a publicly available dataset with the same characteristics as the first one, but with text in English.

It is worth mentioning that the dataset was split into **training**, **validation**, and **test** sets. The training set contains approximately 80% of the images, while the validation set contains 20% of them. The ratio **80:20** is the optimal, since we want variety of training data, but we also don't want our model overfits them. As for the test set, it consists of the last 6 images of the **Specific** dataset.


## Preprocessing
One of the first steps in any machine learning project is data preprocessing, in order to clean and prepare the data for training. In this project, the following preprocessing steps were applied to the images:

* **Crop**: The images were cropped to remove any unnecessary parts, such as borders or margins.
* **Resize**: The images were resized to a fixed size of **1220 x 1220** pixels to ensure that all images have the same dimensions.
* **Normalization**: The pixel values of the images were normalized to the range [0, 1] by dividing them by 255.
* **Greyscale**: The images were converted to greyscale, since the color information is not relevant for the task of text recognition.

## Target Encoding
In order to train the model, the transcript of each image needed to be encoded as a sequence of integers. This is done by creating a mapping between characters and integers, where each character is assigned a unique integer. However, since the transcripts do not contain a fixed length of characters, we needed to pad the sequences to a maximum length, using an extra integer to represent the padding. In this project, the final sequence length was set to a number near to **5000**.

## Model Architecture
The model used in this project is based on the CRNN architecture, which consists of three main components: a convolutional neural network (CNN) for feature extraction, a recurrent neural network (RNN) for sequence modeling, and a connectionist temporal classification (CTC) layer for sequence decoding.

* **CNN**: The role of the CNN is reduce the dimensionality of the input image in order to decrease the noise from the input and extract the most important features. Additionally, as for it's structure, the CNN is composed of several convolutional layers followed by max-pooling layers.

* **RNN**: The RNN ,which consists of a sequence of recurrent layers, takes as input the CNN's output and processes it sequentially in order to find patterns in the data. More specifically, in this project, the **LSTM** (Long Short-Term Memory) recurrent layer was used, which is capable of learning long-term dependencies in the data.

* **CTC**: The **Beam CTC** layer is used to decode the output of the RNN and obtain the final transcript of the image. The CTC layer is responsible for aligning the predicted sequence with the ground truth sequence, taking into account the possibility of repeated characters and the presence of padding.

## Evaluation Metrics

For evaluating the performance of the model, the following metrics were used:

* **Accuracy**: Accuracy is the ratio of correctly predicted instances to the total instances in the dataset. It is a measure of how well the model is performing on the training and validation sets.

* **Precision**: Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. It is a measure of the model's ability to avoid false positives.

```python
Precision = TP / (TP + FP)
```

* **Recall**: Recall measures the proportion of true positive predictions among all actual positives.

```python
Recall = TP / (TP + FN)
```


* **F1 Score**: The F1 score is the harmonic mean of precision and recall, since it takes into account both false positives and false negatives. 

```python
F1 = 2 * (precision * recall) / (precision + recall)
```

* **CER (Character Error Rate)**: The CER is the ratio of the number of character errors to the total number of characters in the ground truth sequence and it is calculated by the formula:

```python
CER = I/N
```

Where:
- I is the number of incorrect characters
- N is the total number of characters 