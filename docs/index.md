# Background

## What is machine learning?

Here is a nice introductory article from [IBM](https://www.ibm.com/topics/machine-learning), one from [Quanta magazine](https://www.quantamagazine.org/what-is-machine-learning-20240708/), one from [UC Berkeley](https://ischoolonline.berkeley.edu/blog/what-is-machine-learning/) which is where the following (highly paraphrased) three ingredients were (to my knowledge) first outlined:

1. A model that takes data and "guesses" a pattern.
2. An error function that tells you how well the model guessed the pattern underlying the data.
3. A way to update the model parameters based on how good or bad the guess is.

In this guided tour, we will mostly focus on supervised learning tasks with increasing complexity. The goal is to mostly implement things by hand to try to understand how everything works together to produce a useful model.
