#!/bin/python

import torch
from torchvision import datasets
def main():
    dataset1 = datasets.MNIST('./data', train=True, download=True)
    dataset2 = datasets.MNIST('./data', train=False, download=True)
if __name__ == '__main__':
    main()
