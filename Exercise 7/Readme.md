
## Exercise Sheet 7

# Network Analysis: Image Classification - Part 2 

In the previous lab, we have implemented CNNs for an image classification task without paying much
attention to the training behavior. Unchecked neural networks may or may not work based on their
initialization. In this lab, we will try to exercise a fine-grain control over the parameters of the network.
For this lab, you will implement the network following architecture and use the CIFAR dataset:

1. conv1: convolution and rectified linear activation (RELU)
2. pool1: max pooling
3. conv2: convolution and rectified linear activation (RELU)
4. conv3: convolution and rectified linear activation (RELU)
5. pool2: max pooling
6. FC1: fully connected layer with rectified linear activation (RELU)
7. FC2: fully connected layer with rectified linear activation (RELU)
8. FC3: fully connected layer with rectified linear activation (RELU)
9. softmaxlayer: final output predictions i.e. classify into one of the ten classes.

NOTE: the kernel size, FC layer hidden units etc. is left up to you to design (bigger FC layers would
lead to additional model complexity).
You will notice that the network is a little bit poorly designed and that is intentional. We would like
the network to be more complex than is needed to overfit to our data. In addition to the increased
complexity of the network, we will also reduce the amount of training data to 1/2 and also remove all
data augmentations/normalizations.
With this network design and data constraint, you are required to get a baseline result to showcase the
model behavior. Pleaseplot both the training and test accuracy and loss for all minibatches
(not just at the end of training). This can be achieved by interleaving a testing step after every 50-
minibatch during model training.
This baseline aims to serve as a straw-man example that we can compare our next steps against.


# Exercise 1: Normalization Effect (CNN) 

Now that we have a weak baseline, we can start to investigate how to improve the model performance.
We will first try to address the data aspect of model improvement.
Improving data:

1. Data Augmentation: the process of artificially ’increasing’ our dataset by adding translation, scaling
    and flipping to the images to fabricate examples for training.
2. Normalization: Normalizing the input data helps remove the dataset artifacts that can cause poor
    model performance.

In this task, you are required to add data augmentation and normalization to the dataset and run the
training. Compare the training performance (accuracy and loss) to the baseline first with only data
augmentation, secondly with only normalization, and lastly with both.Comment on the difference
you observe in the training behavior and the final accuracy as seen on the test set. NOTE:
You should use tensorboard plots.

# Exercise 2: Network Regularization (CNN) 

Having tried to fix the dataset problems, we can have a look at the network itself. Regularization
techniques are useful in learning a generalizable solution and help in avoiding overfitting. For the Deep
Neural Network regularization is generally achieved by using a dropout technique, L1,L2 regularization
among many other techniques. Here we will look at the dropout technique.

- Dropout: Adding dropout layers to a network simply means we stochastically turn off a set number
    of neurons in our Fully connected layer to prevent the model from overfitting on training data.

Similar to the first task, we are interested in seeing how the addition of network regularization can
improve model training behavior and overall performance. You are required to add a dropout layer and
report the training and testing loss and accuracy similar to the exercise above. Comment on the
impact of this modification when compared to the baseline approach.

# Exercise 3: Optimizers (CNN) 

In the last part, you will experiment with two different optimizers i.e. SGD and AdamOptimizer, more
specifically, how robust they are to the initial learning rate. The choices for your initial learning rate
are left up to you. Please compare and contrast the behavior of these two optimizers specifically on how
they react when presented with different learning rates. You should plot the training curves as have been
requested in the exercises above.

