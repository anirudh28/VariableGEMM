# VariableGEMM

This repo delves into enhancing the inference performance of Mixture-of-Experts (MoE) models, a technique leveraging sparsity to achieve high capacities with moderate computational costs. We address the challenge of load-imbalanced computations during dynamic routing within MoE layers, which hampers real-world applications on hardware accelerators like GPUs and TPUs. We integrate existing libraries such as CuBLAS and CuSparse to augment our implementation. Our design includes a MoE layer with 64 experts, each processing variable-sized batches of inputs. We utilize CuBLAS's cublasSGemmGroupedBatched function for batched GEMM tasks with variable-sized entries, transforming inputs into column-major format and segregating them into uniform-sized groups. This allows us to quantify the benefits of batching non-uniform GEMM operations against a baseline implementation. We evaluate our approach across three model sizes with varying numbers of experts and observe superior performance of variable-sized batching over the baseline, confirming the efficiency gains. Additionally, we note that increasing batch size leads to reduced computation time due to enhanced parallelization.

## Steps to run on cuda2.cims.nyu.edu
### One-time setup
1. Pull the cuda-12.4 singularity image for Centos 7 using `singularity`.
    `singularity pull nvidia/cuda:12.4.0-runtime-centos7`
2. Update the image path in `run-cuda-12.4.bash`, it's currently set to `/tmp/cuda-12.4.sif`.
3. Create the `build` directory and setup make.
    ```
    mkdir build
    cd build
    cmake ..
    ```
### Before each run
1. Launch a cuda-12.4 environment using `./run-cuda-12.4.bash`.
2. Go to build directory, `cd build`
3. Run `make` to compile the code
4. Run `./test` to run basic tests.
5. Run `./train_moemlp` to run inference on the MOE model.