---
status: new
---

# Neural networks

In this tutorial, we are going to implement a fully-connected feed-forward neural network from scratch. We will then use this neural network to classify the handwritten digits in [MNIST dataset](https://yann.lecun.com/exdb/mnist/) [LeCun, Cortes, and Burges].

The following resources provide a nice basic introduction to neural networks:

 - ???
 - ???

We'll start by coding the neural network. Once done, we'll turn our attention to the MNIST dataset, in particular, we'll try to get familiar with it, plot a few of the digits, figure out how to transform the images into an input for our neural network, etc.


## The `Network` `#!py class`

To efficiently and cleanly write the necessary code, it is very useful to create a `#!python class`. To read about classes in Python see [realpython.com/python-classes](https://realpython.com/python-classes/) (reading the [Getting Started With Python Classes](https://realpython.com/python-classes/#getting-started-with-python-classes) section will at least get you familiar with the basic concepts).
The class definition and the initialization code (`#!python __init__` method) as well as the definition of the required methods are given just below. Your task is to understand the relevant algorithms and fill in the corresponding methods. Let's go!

!!! danger "Beware"
    The `itertools` module is part of the Python standard library. However, `pairwise` was only added in Python v3.10. Make sure this version requirement is satisfied otherwise see footnote for alternative.

```py hl_lines="28-34 37-41" linenums="1"
class Network:
    def __init__(self, definition: tuple):
        self.definition = definition
        self.loss = 0
        self.layer_ids = range(len(definition))
        self.activations = {}
        self.layers = {}
        for idx in self.layer_ids:
            self.layers[idx] = np.zeros(self.definition[idx])
            self.activations[idx] = np.zeros_like(self.layers[idx])
        # Initialize the weights and biases as dictionaries of np.ndarray
        self.weights = {}
        self.biases = {}
        self.weights_grad = {}
        self.biases_grad = {}
        for idx, (ii, jj) in enumerate(pairwise(definition)):
            self.weights[idx+1] = np.random.rand(ii, jj)
            self.biases[idx+1] = np.zeros(jj)
            self.weights_grad[idx+1] = np.random.rand(ii, jj)
            self.biases_grad[idx+1] = np.zeros(jj)

        # Initialize the input and output vectors (later can generalize to
        # tensors if necessary)
        self.input = np.zeros(definition[0])
        self.output = np.zeros(definition[-1])
        

    def forward(self, features: np.ndarray, targets: np.ndarray):
        """Implements the forward propagation algorithm
        """
        assert self.output.shape == targets.shape
        ...
        self.loss = np.sum(np.square(self.output - targets))
        return None


    def backprop(self, targets: np.ndarray):
        """Implements the backpropagation algorithm
        """ 
        assert self.output.shape == targets.shape
        grad = (targets - self.output)
        ...
        return None


    def compute_loss
        
```

### The forward pass

Propagating the input through the neural network and computing the resulting output is known as _forward propagation_. This algorithm that implements it is described below and closely follows Algorithm 6.3 in [Deep Learning](https://www.deeplearningbook.org/) (chapter 6, page 208) by Goodfellow, Bengio, and Courville `[Goodfellow-et-al-2016]`.

!!! warning "Note"
    The weight matrices are defined differently here than in the book. Specifically, the definition here is related to the one in the book by _matrix transposition_.


----
<pre style="white-space:pre-wrap;padding-left:20px">
<b>algorithm:</b> forward propagation through a fully connected neural network
<b>input:</b> W<sup>(i)</sup>, i∈{1,...,l}; the weights of the model
       b<sup>(i)</sup>, i∈{1,...,l}; the biases of the model
       x, the input (features)
       t, the targets
<b>require:</b> f, activation function

h<sup>(0)</sup> = x
<b>for</b> k=1,...,l <b>do</b>
   a<sup>(k)</sup> = b<sup>(k)</sup> + h<sup>(k-1)</sup>W<sup>(k)</sup>
   h<sup>(k)</sup> = f(a<sup>(k)</sup>)
<b>end for</b>
y = h<sup>(l)</sup>
J = loss(y, t)
</pre>
----

### Backpropagation

What is backpropagation? See this article from [IBM](https://www.ibm.com/think/topics/backpropagation), for example. The [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation) article is quite detailed but also very technical.

The pseudocode below closely follows Algorithm 6.4 in [Goodfellow-et-al-2016].

----
<pre style="white-space:pre-wrap;padding-left:20px">
<b>algorithm:</b> back-propagation
<b>input:</b> W<sup>(i)</sup>, i∈{1,...,l}; the weights of the model
       b<sup>(i)</sup>, i∈{1,...,l}; the biases of the model
       x, the input (features)
       y, the output of the model
       t, the targets
<b>require:</b> f, activation function
         f', derivative of the activation function

g = ∇<sub>y</sub>J (grad_y_loss)
<b>for</b> k=l,l-1,...,1 <b>do</b>
   <span style="color:gray"># ⊙ means element-wise multiplication <b>NOT</b> dot product
   # This can be achieved by ordinary multiplication between two
   # numpy 1-d arrays</span>
   g = g ⊙ f'(a<sup>(k)</sup>)
   <span style="color:gray"># Compute the gradients on the weight matrices and bias vectors</span>
   grad_bk_loss = g
   <span style="color:gray"># ⊗ means outer product, np.outer</span>
   grad_Wk_loss = h<sup>(k-1)</sup> ⊗ g
   <span style="color:gray"># Here W is a matrix and g is a vector so use np.matmul
   # This step propagates the gradient to the layer below (k-1)</span>
   g = W<sup>(k)</sup>g
<b>end for</b>
</pre>
----

The method of the `Network` class to implement is `backprop` (see below).
```py
class Network:
    ...

    def backprop(self, targets: np.ndarray):
        """Implements the backpropagation algorithm
        """ 
        assert self.output.shape == targets.shape
        grad = (targets - self.output)
        ...
        return None
```

## A simple dataset for testing

The (U.S.) National Institute of Standards and Technology (NIST) has a nice collection of datasets for non-linear regression along with 'certified' fit parameters.

The page that lists these datasets is [itl.nist.gov/div898/strd/nls/nls_main.shtml](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml).
You can choose whichever model you like. But to be concrete here, I will take the `Chwirut2` [dataset](https://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml) which is exponential-distributed and described by a 3-parameter model.

Let's also implement the model using the [NIST certified parameters](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/v-chwirut2.shtml). The model is
$$
f(x; \beta) + \epsilon = \frac{\exp(-\beta_1 x)}{\beta_2 + \beta_3 x} + \epsilon\,,
$$
with $\beta_1$, $\beta_2$, and $\beta_3$ given by
```
               Certified              Certified
Parameter      Estimate               Std. Dev. of Est.
  beta(1)      1.6657666537E-01       3.8303286810E-02
  beta(2)      5.1653291286E-03       6.6621605126E-04
  beta(3)      1.2150007096E-02       1.5304234767E-03
```

Here is the Python implementation of this model with the best-fit parameters.
```py
def fcert(x: np.ndarray) -> float:
    
    beta1 = 1.6657666537E-01
    beta2 = 5.1653291286E-03
    beta3 = 1.2150007096E-02

    return np.exp(-beta1*x) / (beta2 + beta3*x)
```

!!! danger "Normalizing the dataset"

    For many reasons, neural networks **do** care about the normalization of the data. In particular, when using the `sigmoid` activation function wich has a range $\in [0,1]$, this would save a lot of unecessary frustration.

