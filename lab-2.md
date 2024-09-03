# Lab-2: Implementing Multilayer Perceptron (MLP)

The goal of this lab is demonstrate how to build a simple MLP model using the basic tensor operations developed in Lab-1.
Alas, instead of autograd, we'll perform manual gradient calculation.
Nevertheless, this lab demonstrates to underlying abstractions of ML systems that help programmers build models out of tensor operators.

## Preliminary: Obtaining the lab code
For this Lab, you will work in the same repository that you've worked on for Lab-1.  Firstly, click on the lab-2 assignment link given in the Campuswire. Then clone your lab2 repository:
```
$ git clone git@github.com:nyu-mlsys-sp24/barenet2-<YourGithubUsername>.git barenet2
```

You'll need to copy three files, `op_mm.cuh`, `op_elemwise.cuh` and `op_reduction.cuh`, from your Lab1's repository to your Lab2's respository. Suppose your Lab-1's repository is `barenet` and your Lab-2's repository is `barenet2`. Then do the following:
```
$ cd barenet2
$ cp ../barenet/src/ops/op_mm.cuh src/ops/
$ cp ../barenet/src/ops/op_elemwise.cuh src/ops/
$ cp ../barenet/src/ops/op_reduction.cuh src/ops/
```

## Understanding Barenet's MLP training loop 

Before you start to write code for Lab-2, it is important to first read the file 
`train_mlp.cu` to understand the barenet's MLP program structure.  We can see that
its main functionality is a training loop implemented by the [`train_and_test`](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L25) function.

In the `train_and_test` function, we first load the MNIST training and test
dataset with the following code:

```
MNIST mnist_train{"../data/MNIST/raw", MNIST::Mode::kTrain};
MNIST mnist_test{"../data/MNIST/raw", MNIST::Mode::kTest};
```

Next, we construct an MLP model by calling the MLP object constructor with
arguments such as the batch size, the input feature dimension size and a list of
sizes representing each layer's output dimension size. By default,  we build a
2-layer model: the first layer's weight tensor shape is 784x16 (input dimension:784, hidden
dimension:16) and the second layer's weight tensor shape is 16x10 (hidden dimension:16, output
dimension:10).  The code for doing so is as follows:

```
MLP<float> mlp{batch_size, MNIST::kImageRows * MNIST::kImageColumns, layer_dims, false};
```

The input dimension is 784 because each MNIST image has 28
(`MNIST:kImageRows`) by 28 (`MNIST:kImageColumns`) pixels, each of which is represented by a float. The output
dimension is 10 because the model needs to classify the image into one of 10 digits (0-9).


You can change the number of layers, batch size, and the hidden dimensions using
[commandline
argumnents](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L91).  


After the MLP is constructed, we initialize its weights with `mlp.init()`.  We then make the simple SGD optimizer with learning rate 0.01.

After the model and the optimizer have been constructed, the training can
begin. Training goes through a fixed number epochs. In each epoch (see function
`do_one_epoch`), we go through the entire training data set sequentially one
batch at a time.  

For each batch, we call `model.forward(input_images, logits)`
to perform the forward computation on a batch of `input_images` and put the
resulting logit values in the output tensor `logits`.  Then, we calculate the
CrossEntropy loss for the batch of data using operator
[`op_cross_entropy_loss`](https://github.com/nyu-mlsys-sp24/barenet/blob/76486c852dd353a968879b4ca37485a579c269bc/src/train_mlp.cu#L66C1-L67C1).
The operator `op_cross_entropy_loss` is a manually fused operator that
calculates the loss given the logits tensor (computed by `model.forward`), the
`targets` tensor containing the batch's training labels.  Additionally, the
operator also calculates the gradients of the logits and put them in the
`d_logits` output tensor. With the `d_logits` gradients, we can start the rest of the
backward computation by calling `model.backward(input_images, d_logits,
d_input_images).` to compute the gradients for each layer's parameter tensors. 
Finally, we take a gradient descend step to update the
model parameters using `sgd.step()`. 

We keep track of the statistics of the
loss and prediction accuracy and report them at the end of an
epoch.  Once training is done (after a fixed number of epochs), we calculate how the model performs 
on the testing datatest. Specifically, we go through the testing dataset one batch at a time just
like training, except that there is no need for backward propagation and SGD
update.

## Training MLP 

### Training MLP in PyTorch

You will find it very helpful to see how the same MLP training can be done in
PyTorch, which is provided for you in ``mnist_mlp.ipynb``. Launch Jupyter
Notebook to run it. Note that you do not have to run it on GPU; running it on
CPU is fine (running it on your own laptop also works).  After 10 epochs of
training, you should get a training accuracy and testing accuracy $>92\%$.  To get
full marks on this lab, your barenet should get a training and testing accuracy
$>92\%$ as well.

### Training MLP in barenet
To train MLP in barenet, we need to first get the MNIST dataset.  There are two ways to do it: One, you can run the script `mnist_mlp.ipnb` on your cloned lab repository on HPC, which will download and save the MNIST dataset. Alternatively, if you do not want to 
run `mnist_mlp.ipnb` on HPC, you can download the MNIST dataset using the following command.

```
$ cd barenet2
$ python download.py
```

You should see the subdirectory named `data/MNIST/raw` with all the
MNIST training and test data files:
```
$ ls data/MNIST/raw
t10k-images-idx3-ubyte		t10k-labels-idx1-ubyte		train-images-idx3-ubyte		train-labels-idx1-ubyte
t10k-images-idx3-ubyte.gz	t10k-labels-idx1-ubyte.gz	train-images-idx3-ubyte.gz	train-labels-idx1-ubyte.gz
```

Next, compile the code. The compilation procedure is the same as that for
[Lab-1](https://github.com/nyu-mlsys-sp24/barenet/blob/master/lab-1.md#compilation).


After finishing compilation, do MLP training by typing the following:
```
$ cd build
$ ./train_mlp  
```

An example output from a working lab is shown below:
```
# of training datapoints=60000 # of test datapoints= 10000 feature size=784
training datapoints mean=0.131136
TRAINING epoch=0 loss=0.893307 accuracy=0.787633 num_batches=1875
TRAINING epoch=1 loss=0.390445 accuracy=0.891333 num_batches=1875
TRAINING epoch=2 loss=0.338284 accuracy=0.903617 num_batches=1875
TRAINING epoch=3 loss=0.313121 accuracy=0.910933 num_batches=1875
TRAINING epoch=4 loss=0.295669 accuracy=0.915917 num_batches=1875
TRAINING epoch=5 loss=0.282319 accuracy=0.919767 num_batches=1875
TRAINING epoch=6 loss=0.271276 accuracy=0.922933 num_batches=1875
TRAINING epoch=7 loss=0.261205 accuracy=0.924933 num_batches=1875
TRAINING epoch=8 loss=0.251797 accuracy=0.927883 num_batches=1875
TRAINING epoch=9 loss=0.243037 accuracy=0.93055 num_batches=1875
TEST epoch=0 loss=0.241526 accuracy=0.928986 num_batches=312
```

## Suggested Steps for Completing Lab-2 

Although you can complete Lab 2 however you like, here's our suggested plan.

### Step-1: Complete the forward path

The MLP model consists of several linear layers
interspersed by non-linear activation functions.
The file `modules/linear.cuh` implements the Linear Layer. The code is missing the
implementation of the ``forward`` function. Complete this function.

The file `modules/mlp.cuh` implements the MLP model using Linear Layers and activation functions. The code is missing the 
implementation for its ``forward`` function. Complete this function.

In preparation for the backward path, we must save the activation tensors 
(the output of each linear layer).  In barenet, we save them in the MLP object
as a vector of tensors ``MLP::activ``. 

### Step-2: Implement operator `op_cross_entropy_loss`

Given a batch of input images, the forward path of MLP calculates the logits
tensor whose shape is $b\times c$ where $b$ is the batch size and $c$ is the
number of image classes (10 for MNIST).  To predict the class label, we tranform logits into 
probabilities using [Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html).
Given the model's predicted probabilities and the actual class labels given by the 
``targets`` tensor, we compute loss using the [cross
entropy function](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html).

We could implement `op_cross_entropy_loss` using a series of primitive
operators (e.g. using ``op_add`` ``op_multiply`` ``op_sum``, ``op_exp`` and
``op_log`` etc.) and later individually back-prop through each of them for the
backward path.  However, doing so is efficient as the approach we ask
you to take (which is also the approach taken by PyTorch).  Essentially, our plan is to 
to "fuse" the forward and backward computation from logits to cross entropy loss into a
single operator to make it more efficient. This operator,  `op_cross_entropy_loss`,
takes two input arguments: the logits tensor for a batch of images and the batch's corresponding target label
tensor. It returns the average loss of the batch as well as the gradients of the
logits tensor. We'll use our calculus knowledge to derive these gradients, and
doing so is much more efficient.

Let $X$ represent the MLP's model output (aka logits) for a single datapoint (at index $d$ of the batch).
$X$ is a vector of size $c$ (where $c=10$ for MNIST).  We calculate
$P=Softmax(X)$, in which the probability of class $i$ is calculated as:
$$p_i = \frac{e^{x_i}}{\sum_{j=1}^{c}e^{x_j}}$$

The cross entropy loss for this datapoint is $$l_d = -\sum_{i=1}^{c}y_i*log(p_i)$$
where $y_i=1$ if $i$ is the target class label for this datapoint and $y_i=0$
otherwise. To simplify this expression, suppose the target label for this datapoint is $t$, then loss is:
$l_d = -log(p_t)$. 

The average loss for the entire batch of $b$ datapoints is:
$loss = \frac{1}{b}\sum_{d=0}^{b-1}{l_d}$.

Using Calculus knowledge, we compute the gradients: $\frac{d l_d}{d x_i}$. We need to distinguish two 
cases. For $i=t$, we calculate 
$$
\begin{align*}
\frac{d l_d}{d x_i} &= \frac{d}{d x_i}[-log\frac{e^{x_i}}{\sum e_{x_j}}]\\
&= \frac{d}{d x_i}[log(\sum e^{x_j})-x_i]\\
&= (\frac{d}{d x_i}[log \sum e^{x_j}]) -1 \\
&=\frac{e^{x_i}}{\sum e^{x_j}} -1\\
&= p_i -1\\
\end{align*}
$$

 For $i\neq t$, we calculate: 
 $$
\begin{align*}
\frac{d l_d}{d x_i} &= \frac{d}{d x_i}[-log\frac{e^{x_t}}{\sum e_{x_j}}]\\
&= \frac{d}{d x_i}[log(\sum e^{x_j})-x_t]\\
&= p_i\\
\end{align*}
$$
 
 The final gradients of the logits tensor for the averaged loss of a batch is: $\frac{d loss}{d x_i} = \frac{d loss}{d l_d}\cdot \frac{d l_d}{d x_i} = \frac{1}{b}\cdot \frac{d l_d}{d x_i}$.

 There is one more thing to note. In practice, we compute a safe form of Softmax to guard against 
 overflow or underflow when computing with exponentiated numbers because the range and precision 
 of float point numbers are limited. All DL frameworks including TensorFlow and PyTorch use this safe version 
 of Softmax computation:
$$p_i = \frac{e^{x_i-x_{max}}}{\sum_{j=1}^{c}e^{x_j-x_{max}}}$$
where $x_{max} = max(X)$. Make sure you use this safe version of Softmax to calculate $p_i$.

We have added a unit test in file `test.cu` to test the correctness of your `op_cross_entropy_loss` implementation.
When the unit test passes, the output looks like this:
```
$ make
$ ./test
slice passed...
op_add passed...
op_multiply passed...
op_sgd passed...
matmul passed...
op_sum passed...
op_cross_entropy_loss passed...
All tests completed successfully!
```

### Step-3: Complete the backward path

Now, you are ready to implement the backward path.

First, complete the missing implementation of the `backward` function for the Linear layer in the file `modules/linear.cuh`.
Next, complete the missing implementation of the `backward` function for the MLP model in the file `modules/mlp.cuh`.
You may notice that the MLP object also keeps a vector of gradient tensors
on the activation tensors, ``MLP:d_activ``.  Strictly speaking, it is unnecessary to
keep these gradient tensors as MLP's member variables. This is because, on the
backward path, we can calculate a gradient tensor for the activation, use it immediately (as input
to the previous layer's backward step) and discard it afterwards. However,
doing so results in frequent memory allocation/deallocation for these gradient
tensors, resulting in overhead for each training step.  Hence, we allocate a
fixed set of gradient tensors ``MLP:d_activ``. In your implementation of `backward`, 
you will rewrite these activation gradient tensors.

## Hand-in procedure

As in Lab-1, you should save your progress frequently by doing `git commit` followed by `git push origin master`.

To hand in your lab, first commit all of your modifications by following the instructions in the section on [Saving your progress](#Saving-your-progress). Second, make a tag to mark the latest commit point as your submittion for Lab1. Do so by typing the following:
```
$ git tag -a lab2 -m "submit lab2"
```

Finally, push your commit and your tag to Github by typing the following
```
$ git push --tags origin master
```


