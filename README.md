# signs
Training a CNN from scratch to classify the hand-digit signs dataset. This model will use Keras' flexible Functional API to build a ConvNet that can differentiate between 6 sign language digits - the numbers 0, 1, 2, 3, 4, and 5. The dataset consists of 1080 64x64x3 test images in 6 classes. The accuracy of the small model from scratch is around 99.17-100%. This model uses techniques such as reducing learning rate, early stopping, and data augumentation to increase the accuracy. Run time on a GPU is around 177 seconds.

![image](https://github.com/carab9/signs/blob/main/signs.png?raw=true)

## Datasets
Hand signs datasets either in the local directory, cifar-10-datasets, or on the AWS S3 bucket.

## Requirements
Python, Tensorflow, Keras, Jupyter Notebook, AWS S3, AWS Sagemaker.

## Technical Skills
Tensorflow, Keras APIs, CNN architecture, regularization techniques such as reducing learning rate, early stopping, and data augumentation, AWS S3, and AWS Sagemaker.

## Results
![image](https://github.com/carab9/signs/blob/main/signs_loss.png?raw=true)

![image](https://github.com/carab9/signs/blob/main/signs_accuracy.png?raw=true)
