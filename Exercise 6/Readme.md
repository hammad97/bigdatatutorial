
## Exercise Sheet 6

# Exercise 1: PyTorch Network Analysis 

For this first task, you are required to implement LeNet (see Annex) for the following datasets for image
classification:

- MNIST
- CIFAR

Both datasets are available in the PyTorch Datasets library and you may use the library to service
the model.Note: LeNet was originally written for MNIST dataset which has 1-Channel black and white
images, so you would need to keep that in mind when you want to implement it for CIFAR10, which is
a 3-Channel RGB image dataset. Furthermore, the image size for the two datasets is different, that is
something to keep in mind as well when adapting LeNet for CIFAR10.

Some configurations for you to try when training MNIST and CIFAR are as follows:

- Learning rate [0.1, 0.01, 0.001]
- Optimizer [SGD, ADAM, RMSPROP]

Note that you do not necessarily need to provide results for all combinations of these configurations.
A valid approach would be to fix the choice of the Optimizer and try different learning rates for that
Optimizer only.

One thing that is essential to learn when training network models is usingTensorboard with Py-
Torch. Tensorboard enables us to visualize the network performance in an effective manner. Please
report the Train/Testaccuracyand Train/Testlossusing Tensorboard. Refer to the Annex section for
useful tips.


# Exercise 2: Custom Task 

In Exercise 1 we implemented a well-known network for image classification. Building upon that learning,
we will now extend the network to an image regression setting. Using the MNIST dataset, we will design a
network that will consume (N,K,C,W,H) batches of images where N,K,C,W,H are batch size, number
of images to sum, channels, width and height respectively. The output of the network will be the sum
of the number in the K many images. For instance, if we haveK= 3 and the three images contain the
digits 1, 5 and 3, then the output of the network should be 9.
To setup this network, you need to write a custom dataset that will extend the basic dataset module to
return a batch-shaped N,K,C,W,H data format instead of the typical N,C,W,H data shape. You should
choose K many random images to populate the batch instances. You will need to adapt you network
to consume this extended dataset dimension. Use thex=x.view(N∗K, C, W, H) trick discussed in
the lab session. Your ground truth target during training will be the sum of the K many images. This
can be calculated by summing the integer labels already associated with the images. The design for the
network output (predictions) is left up to you, the only requirement here is that it should be a numeric
value. Keeping in mind that we are no linger looking at a classification problem but rather a regression
problem, our loss should also be modified to reflect that.HINT: Regular Maintenance Saves Energy.
For this task, you are to report:

- Train/Test Error on Tensorboard
- 10 sets of test images (each set containing K many images) and the output from your trained
    network (final value that your model outputs)

Please display the output from your trained network onlyafteryour model is done training, having
a visual output will help you get in the habit of visually inspecting your network behavior.



