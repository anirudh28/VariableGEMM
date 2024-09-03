#include <getopt.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include "modules/MOEmlp.cuh"
#include "modules/linear.cuh"
#include "utils/tensor.cuh"

#include "utils/dataset_mnist.hh"
#include "ops/op_elemwise.cuh"
#include "ops/op_cross_entropy.cuh"

unsigned long long randgen_seed = 1;

static bool on_gpu = true;

int correct(const Tensor<float> &logits, const Tensor<char> &targets) 
{
    assert(targets.w == 1);
    Tensor<int> predictions{targets.h, targets.w, on_gpu};
    op_argmax(logits, predictions);
    Tensor<int> correct_preds{targets.h, targets.w, on_gpu};
    op_equal(predictions, targets, correct_preds);
    Tensor<int> sum_correct{1,1, on_gpu};
    op_sum(correct_preds, sum_correct);
    if (on_gpu) {
        auto tmp = sum_correct.toHost();
        return Index(tmp, 0, 0);
    }
    return Index(sum_correct, 0, 0);
}

void generate_random_moe_splits(std::vector<int>& splits, int total_batch_size, int num_splits) {
    std::mt19937 generator(randgen_seed);  
    std::vector<int> temp_splits(num_splits, 0);
    int sum = 0;

    for (int i = 0; i < num_splits - 1; ++i) {
        std::uniform_int_distribution<int> distribution(1, total_batch_size - sum - (num_splits - i));
        temp_splits[i] = distribution(generator);
        sum += temp_splits[i];
    }
    // Ensuring the splits sum to total_batch_size
    temp_splits[num_splits - 1] = total_batch_size - sum; 
    splits = std::move(temp_splits);
}


long long test(int batch_size, int hidden_dim, int n_layers)
{
    MNIST mnist_test{"../data/MNIST/raw", MNIST::Mode::kTest};
    // std::cout << "# of test datapoints= " << mnist_test.images.h << " feature size=" << mnist_test.images.w << std::endl;

    auto test_images = mnist_test.images;
    auto test_targets = mnist_test.targets;
    

    int expand = 10;
    int original_size = test_images.h;
    int new_size = original_size * expand;

    Tensor<float> expanded_images{new_size, test_images.w, false};
    Tensor<char> expanded_targets{new_size, test_targets.w, false};
    
    for(int i=0; i < expand; i++){
        for(int j = 0; j < original_size; j++){

            for(int k = 0; k < test_images.w; k++){
                Index(expanded_images, i*original_size + j, k) = Index(test_images, j, k);
            }

            Index(expanded_targets, i*original_size + j, 0) = Index(test_targets, j, 0);
        }
    }

    std::cout << "# of test datapoints= " << expanded_images.h << " feature size=" << expanded_images.w << std::endl;

    if(on_gpu){
        test_images = test_images.toDevice();
        test_targets = test_targets.toDevice();
        expanded_images = expanded_images.toDevice();
        expanded_targets = expanded_targets.toDevice();
    }
    
    std::vector<int> layer_dims;
    for (int i = 0; i < n_layers - 1; i++)
    {
        layer_dims.push_back(hidden_dim);
    }
    layer_dims.push_back(10); // last layer's out dimension is always 10 (# of digits)

    // std::vector<int> moe_batch_splits{8, 6, 10, 8}; 
    std::vector<int> moe_batch_splits;
    generate_random_moe_splits(moe_batch_splits, batch_size, 4);

    MOEMLP<float> moeMLP{batch_size, MNIST::kImageRows * MNIST::kImageColumns, layer_dims, moe_batch_splits, on_gpu};
    moeMLP.init();

    Tensor<float> logits{batch_size, 10, on_gpu};
    int num_batches = 0, total_correct = 0;

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
		
    for (int b = 0; b < expanded_images.h / batch_size; b++)
    {
        if ((b + 1) * batch_size > expanded_images.h)
        {
            break;
        }
        num_batches++;
        Tensor<float> b_images = expanded_images.slice(b * batch_size, (b + 1) * batch_size, 0, expanded_images.w);
        Tensor<char> b_targets = expanded_targets.slice(b * batch_size, (b + 1) * batch_size, 0, expanded_targets.w);
		moeMLP.forward(b_images, logits, on_gpu);
        total_correct += correct(logits, b_targets);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "TEST accuracy=" << total_correct / (float)(num_batches * batch_size)
              << " num_batches=" << num_batches
              << " Inference Time: " << duration.count() << " ms" << std::endl;
    
    return duration.count();
}


int main(int argc, char *argv[])
{
    int hidden_dim = 16;
    int n_layers = 2;
    int batch_size = 32;

    for (;;)
    {
        switch (getopt(argc, argv, "s:g:h:l:b:"))
        {
        case 's':
            randgen_seed = atoll(optarg);
            continue;
        case 'g':
            on_gpu = atoi(optarg)?true:false;
            continue;
        case 'h':
            hidden_dim = atoi(optarg);
            continue;
        case 'l':
            n_layers = atoi(optarg);
            continue;
        case 'b':
            batch_size = atoi(optarg);
            continue;
        case -1:
            break;
        }
        break;
    }
    // test(batch_size, hidden_dim, n_layers);
    std::vector<int> various_hidden_dims = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::ofstream file("inference_times.csv");
    file << "Hidden Layer Size,Time (ms)\n";

    for(int size: various_hidden_dims){
        long long time = test(batch_size, size, n_layers);
        file << size << "," << time << "\n";
    }
    file.close();
    return 0;
}
