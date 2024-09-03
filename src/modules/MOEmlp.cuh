#pragma once
#include "modules/linear.cuh"
#include "modules/MOElinear.cuh"
template <typename T>
class MOEMLP
{
private:
    MOELinearLayer<T> moeLayer; //Adding a single MOE linear layer (Can add multiple in future)
    std::vector<LinearLayer<T>> layers;
    std::vector<int> layer_dims;
    std::vector<Tensor<T>> activ;
    std::vector<Tensor<T>> activf;
    int batch_size;
    int in_dim;
    std::vector<int> moe_batch_splits;

public:
    MOEMLP(int batch_size_, int in_dim_, std::vector<int> layer_dims_, std::vector<int> moe_batch_splits_, bool gpu)
        : batch_size(batch_size_), in_dim(in_dim_), layer_dims(layer_dims_), moe_batch_splits(moe_batch_splits_), moeLayer()
    {
        moeLayer = MOELinearLayer<T>(in_dim, layer_dims[0], moe_batch_splits, gpu);
        for (int i = 0; i < layer_dims.size(); i++)
        {
            if (i == 0)
            {
                // layers.emplace_back(in_dim, layer_dims[i], gpu);
                layers.emplace_back(layer_dims[0], layer_dims[i], gpu);
            }
            else
            {
                layers.emplace_back(layer_dims[i - 1], layer_dims[i], gpu);
            }
        }
        // make all the activation tensors
        activ.reserve(layer_dims.size() - 1);
        activf.reserve(layer_dims.size() -1);
        for (int i = 0; i < layer_dims.size() - 1; i++)
        {
            activ.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
            activf.emplace_back(Tensor<T>(batch_size, layer_dims[i], gpu));
        }
    }

    void init() {
        moeLayer.init();
        for (int i = 0; i < layers.size(); i++) {
            layers[i].init_uniform();
        }
    }

    //This function peforms the forward operation of a MLP model with a MOE layer before its first layer
    //Specifically, it should call the forward oepration of each linear layer 
    //Except for the last layer, it should invoke Relu activation after each layer.
    void forward(const Tensor<T> &in, Tensor<T> &out, bool gpu)
    {
        
        Tensor<T> temp(in.h, layer_dims[0], gpu);
        moeLayer.forward(in, temp);
        for(int i=0; i < layers.size(); i++){
            if(i < layers.size()-1){
                layers[i].forward(temp, activf[i]);
                op_relu(activf[i], activ[i]);
                temp = activ[i];
            }
            else{
                //For the last layer
                layers[i].forward(temp, out);
            }
        }
    }
};
