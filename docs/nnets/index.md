
# Neural networks

In this tutorial, we are going to implement a fully-connected feed-forward neural network from scratch. We will then use this neural network to classify the handwritten digits in [MNIST dataset](https://yann.lecun.com/exdb/mnist/) [LeCun, Cortes, and Burges].

The following resources provide a nice basic introduction to neural networks:

 - [https://www.ibm.com/topics/neural-networks](https://www.ibm.com/topics/neural-networks)
 - [https://en.wikipedia.org/wiki/Neural_network_(machine_learning)](https://en.wikipedia.org/wiki/Neural_network_%28machine_learning%29)
 - [https://realpython.com/python-ai-neural-network/](https://realpython.com/python-ai-neural-network/)
 - You can also read Chris Bishop's new book online, see [https://www.bishopbook.com/](https://www.bishopbook.com/). Chapter 4 discusses single layer neural networks.
   - And his _classic_ PRML book can also be downloaded from Microsoft for free here [https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/). Chapter 5 dicusses neural networks.

We'll start by coding the neural network. Once done, we'll turn our attention to the MNIST dataset, in particular, we'll try to get familiar with it, plot a few of the digits, figure out how to transform the images into an input for our neural network, etc.

