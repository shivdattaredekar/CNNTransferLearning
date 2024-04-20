# Transfer Learning in CNN with LeNet, AlexNet, and VGGNet

This project explores the implementation of transfer learning in Convolutional Neural Networks (CNN) using three popular architectures: LeNet, AlexNet, and VGGNet. The transfer learning technique leverages pre-trained models on large datasets to improve the performance of models trained on smaller datasets. In this project, we apply transfer learning to two distinct datasets: the Dog-Cat dataset and the Flowers 17 dataset.

## Introduction

Transfer learning involves using knowledge gained while solving one problem and applying it to a different but related problem. In the context of deep learning, transfer learning typically involves using a pre-trained model on a large dataset (such as ImageNet) and fine-tuning it on a smaller dataset with a similar task.

## Datasets

- **Dog-Cat Dataset**
  - The Dog-Cat dataset consists of images of dogs and cats.
  - It is commonly used for binary classification tasks in computer vision.
  - The dataset is divided into training and testing sets.

- **Flowers 17 Dataset**
  - The Flowers 17 dataset contains images of 17 different categories of flowers.
  - Each category consists of a variable number of images.
  - It is often used for multi-class classification tasks in computer vision.

## Models

- **LeNet**
  - LeNet is one of the earliest CNN architectures, developed by Yann LeCun et al. in 1998.
  - It consists of convolutional and pooling layers followed by fully connected layers.
  - LeNet is relatively simple compared to modern architectures but can still achieve decent performance on certain tasks.

- **AlexNet**
  - AlexNet, introduced by Alex Krizhevsky et al. in 2012, was the breakthrough model that popularized CNNs in computer vision.
  - It comprises several convolutional and max-pooling layers followed by fully connected layers.
  - AlexNet demonstrated significant improvements in image classification accuracy over previous methods.

- **VGGNet**
  - VGGNet, proposed by Simonyan and Zisserman in 2014, is known for its simplicity and uniform architecture.
  - It consists of a series of convolutional layers with small 3x3 filters, followed by max-pooling layers and fully connected layers.
  - VGGNet achieved state-of-the-art performance on the ImageNet dataset at the time of its introduction.

## Implementation

- Each CNN architecture (LeNet, AlexNet, VGGNet) is first initialized with weights pretrained on the ImageNet dataset.
- The pre-trained models are then fine-tuned on the respective training datasets (Dog-Cat and Flowers 17) to adapt them to the specific classification tasks.
- We evaluate the performance of each model using metrics such as accuracy, precision, recall, and F1-score on the testing datasets.

## Results
We present the results of our experiments, including accuracy and other relevant metrics, in the "results" directory.

## Conclusion
This project demonstrates the effectiveness of transfer learning in improving the performance of CNN models on small datasets. By leveraging pre-trained models and fine-tuning them on task-specific data, we achieve competitive results on both the Dog-Cat and Flowers 17 datasets.

## Contributors
  - Shivdatta Redekar
  - shivdattaredekar@gmail.com
  - LinkedIn = https://www.linkedin.com/in/shivdatta-redekar-93ab1511a

## License
  - This project is licensed under the MIT License
