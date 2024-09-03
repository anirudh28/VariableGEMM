#pragma once

#include "utils/tensor.cuh"

#define ELEMWISE_BLOCK_DIM 32

template <typename T>
class MaxAccumFunc
{
public:
    //This function compares input x with the current accumulated maximum value stored in accum
    //If x is bigger than accum, stores x in accum and stores x's index (ind_x) to ind_accum
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      //Lab-1: add your code here
      if(x > accum){
        accum = x;
        ind_accum = ind_x;
      }
    }
};

template <typename T>
class SumAccumFunc
{
public:
    //This function adds input x to the current accumulated sum value stored in accum
    //The accumu's value is updated (to add x).  The ind_x and ind_accum arguments are not used. 
    __host__ __device__ void operator()(const T &x, const int &ind_x, T &accum, int &ind_accum)
    {
      //Lab-1: add your code here
      accum += x;
    }
};

//This kernel function performs column-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_colwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    //Lab-1: add your code here
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < in.h){
        T accum = Index(in, row, 0);
        int ind_accum = 0;
        for(int col=1; col < in.w; col++){
            T x = Index(in, row, col);
            f(x, col, accum, ind_accum);
        }

        if(get_index){
            Index(out_index, row, 0) = ind_accum;
        }
        else{
            Index(out, row, 0) = accum;
        }
    }

}

//This kernel function performs row-wise reduction of the "in" tensor and stores the result in "out" tensor.
//If "get_index" is true, then the index accumulator values are stored in "out_index" and "out" is not touched.
template <typename OpFunc, typename T>
__global__ void op_reduction_kernel_rowwise(OpFunc f, Tensor<T> in, Tensor<T> out, Tensor<int> out_index, bool get_index)
{
    //Lab-1: add your code here
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < in.w){
        T accum = Index(in,0,col);
        int ind_accum = 0;
        for(int row=1; row< in.h; row++){
            T x = Index(in, row, col);
            f(x, row, accum, ind_accum);
        }

        if(get_index){
            Index(out_index, 0, col) = ind_accum;
        }
        else{
            Index(out, 0, col) = accum;
        }
    }
    
}    

template <typename OpFunc, typename T>
void op_reduction_gpu(OpFunc f, const Tensor<T> &in, Tensor<T> &out, Tensor<int> &out_index, bool get_index = false)
{
    int out_h = out.h;
    if (!get_index) {
        assert((out.h == 1 && in.w == out.w) || (out.w == 1 && in.h == out.h));
    } else {
        out_h = out_index.h;
        assert((out_index.h == 1 && in.w == out_index.w) || (out_index.w == 1 && in.h == out_index.h));
    }
    // dim3 blockSize(ELEMWISE_BLOCK_DIM, ELEMWISE_BLOCK_DIM);
    // dim3 numBlocks((in.w + ELEMWISE_BLOCK_DIM -1)/ELEMWISE_BLOCK_DIM, (in.h + ELEMWISE_BLOCK_DIM -1)/ELEMWISE_BLOCK_DIM);
    if (in.h > out_h)
    {
      //Lab-1: add your code here to launch op_reduction_kernel_rowwise
      //delete assert(0) when you are finished
    //   assert(0);
        dim3 blockSize(ELEMWISE_BLOCK_DIM);
        dim3 numBlocks((in.w + ELEMWISE_BLOCK_DIM -1)/ELEMWISE_BLOCK_DIM);
        op_reduction_kernel_rowwise<<<blockSize, numBlocks>>>(f, in, out, out_index, get_index);
    }
    else
    {
      //Lab-1: add your code here to launch op_reduction_kernel_colwise
      //delete assert(0) when you are finished
    //   assert(0);
        dim3 blockSize(1,ELEMWISE_BLOCK_DIM);
        dim3 numBlocks(1,(in.h + ELEMWISE_BLOCK_DIM -1)/ELEMWISE_BLOCK_DIM);
        op_reduction_kernel_colwise<<<blockSize, numBlocks>>>(f, in, out, out_index, get_index);
    }
}


template <typename T>
void op_sum(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    SumAccumFunc<T> f;
    if (in.on_device && out.on_device) {
        op_reduction_gpu(f, in, out, out_index, false);
    } else
        assert(0);
}

template <typename T>
void op_argmax(const Tensor<T> &in, Tensor<int> &out_index)
{
    Tensor<T> out;
    MaxAccumFunc<T> f;
    if (in.on_device && out_index.on_device) {
        op_reduction_gpu(f, in, out, out_index, true);
    } else
        assert(0);
}


template <typename T>
void op_argmax2(const Tensor<T> &in, Tensor<T> &out)
{
    Tensor<int> out_index;
    MaxAccumFunc<T> f;
    if (in.on_device && out.on_device) {
        op_reduction_gpu(f, in, out, out_index, false);
    } else
        assert(0);
}
