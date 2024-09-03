# Lab-1: Implementing basic tensor operators on GPUs.

The goal of this lab is two-fold. One, we want to demonstrate the abstraction of tensors and tensor operations.
Two, we want you to become familiar with GPU programming so as to implement the tensor operations.

## Preliminary: Obtaining the lab code
Follow the instruction given on Campuswire to create your github repository containing the lab skeleton files.
Then, on a HPC machine, type this:
```
git clone git@github.com:nyu-mlsys-sp24/barenet-<YourGithubUsername>.git barenet
```

### Compilation
While you are writing the code and fixing the compilation errors, you do not need a GPU.  Any CPU machine on the HPC cluster 
should be sufficient for compilation as they have `nvcc` (the CUDA compiler) installed. 

To compile, do the following in the lab's directory `barenet`:
```
$ cd barenet
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Our lab uses the [cmake](https://cmake.org/cmake/help/latest/index.html) tool
to generate a Makefile for the project. Once the Makefile is generated, we can use
the `make` tool to compile our code.

## Correctness
Correctness will constitute 70% of Lab-1's score. 
We will evaluate the correctness of your Lab-1 using a simple unit test file `test.cu`.  You need a GPU in order to run the test! 
Once you finish compilation, 
run the unit test as follows:

```
$ ./test
```

If your passed the unit test, the output will look like this:
```
$./test
slice passed...
op_add passed...
op_multiply passed...
op_sgd passed...
matmul passed...
op_sum passed...
All tests completed successfully!
```

## Performance
Performance will constitute 30% of Lab-1's score. We will compare your kernel's performance with those of the instructor's 
own basic implementation.

## Lab-1 instructions

In Lab-1, you will complete the necessary code in `op_elemwise.cuh`, `op_mm.cuh` and `op_reduction.cuh` and pass 
simple unit tests. Your implementation should also be performant in order to get good performance scores.

You should first carefully read through the code in `utils/tensor.cuh`. This file defines our tensor abstraction. 
Here are the things you should know about our tensors:

- The tensor is templated so that we can have different data types for the elements stored in it.

- The tensor's internal buffer is ref-counted using C++ smart pointer and its corresponding memory storage is automatically freed when there are no more references to it.
In other words, you do not need to worry about needing to free the memory.

- For our labs, the tensor is *always* 2-dimensional, with the first dimension named `h` (height) and second dimension named `w` (width).
A row vector has `h=1` and a column vector has `w=1`.

- The Macro `Index(t, i,j)` will  be handy for accessing the element at [i,j] coordinate of tensor t. The Macro `IndexOutOfBound(t, i, j)` will be handy for testing whether [i,j] is out of bound for tensor t.


Next, complete the necessary functions  in `op_elemwise.cuh`, `op_mm.cuh` and `op_reduction.cuh`, in the given order.
Read through the code skeleton, and fill in your code whenever you see a comment that says `Lab-1: please add your code here`.
After finishing each file, you should be able to pass a portion of the unit test. Debug and complete that portion of the unit test 
before moving on to the next lab file.

## Saving your progress

You want to save whatever progress you've made on the lab and back it up frequently so that losing your laptop does not result in the loss of your lab work.  To do so, you commit your file modifications so far and push those commits (aka back them up) to your remote respository on Github.  You do so by typing the following:
```
$ git commit -am "Some meaningful commit message"
$ git push origin master
```

Do the above frequently while you work on the lab.  However, it is generally frowned upon to commit a change that does not compile.

## Hand-in procedure

To hand in your lab, first commit all of your modifications by following the instructions in the section on [Saving your progress](#Saving-your-progress). Second, make a tag to mark the latest commit point as your submittion for Lab1. Do so by typing the following:
```
$ git tag -a lab1 -m "submit lab1"
```

Finally, push your commit and your tag to Github by typing the following
```
$ git push --tags origin master
```

You should double check that your commit and your tag is correctly pushed to the Github by double checking 
on the github webpage. See the Screenshot below as an example ![](https://news.cs.nyu.edu/~jinyang/GithubScreenshot.jpg)



That's it.  **Please do not delete or modify your tag after the Lab submission date. We'll make a copy of the tagged commit from your Github repository immediately after the Lab submission date has passed.**  After you've tagged, you can continue to commit new changes and push your new commits to Github as you move on to doing Lab-2.


