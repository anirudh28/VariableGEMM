#pragma once
#include "modules/linear.cuh"
#include "modules/param.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_sdmm.cuh"
#include "ops/op_reduction.cuh"

#include <algorithm>
#include <vector>

template <typename T> class MOELinearLayer {
private:
  std::vector<LinearLayer<T>> experts;
  std::vector<int> batch_splits;
  // std::vector<cudaStream_t> streams;
  int in_dim;
  int out_dim;

public:
  MOELinearLayer() : in_dim(0), out_dim(0) {}
  MOELinearLayer(int in_dim_, int out_dim_, std::vector<int> &batch_splits_,
                 bool gpu)
      : in_dim(in_dim_), out_dim(out_dim_), batch_splits(batch_splits_) {
    for (int i = 0; i < batch_splits.size(); i++) {
      experts.emplace_back(in_dim, out_dim, gpu);
    }
  }

  void init() {
    for (auto &expert : experts) {
      expert.init_uniform();
    }
  }

  void forward(const Tensor<T> &x, Tensor<T> &y) {
    assert(x.h == y.h && y.w == out_dim);
    int start = 0;
    // read flag from args
    int flag = 1;
		if (flag == 0) {
			for (int i = 0; i < experts.size(); i++) {
        int batch_size = batch_splits[i];
        Tensor<T> x_sub = x.slice(start, start + batch_size, 0, in_dim);
        Tensor<T> y_sub = y.slice(start, start + batch_size, 0, out_dim);
        experts[i].forward(x_sub, y_sub);
        start += batch_size;
      }
		}
    else if (flag == 1 || flag == 2) {
      std::vector<Tensor<T>> x_batched;
      std::vector<Tensor<T>> y_batched;
      std::vector<Tensor<T>> w_batched;
      std::vector<Tensor<T>> b_batched;
			for (int i = 0; i < experts.size(); i++) {
        int batch_size = batch_splits[i];
        x_batched.push_back(x.slice(start, start + batch_size, 0, in_dim));
        y_batched.push_back(y.slice(start, start + batch_size, 0, out_dim));
        w_batched.push_back(experts[i].w.t);
        b_batched.push_back(experts[i].b.t);
        start += batch_size;
      }
			if (flag == 1) {
      	batched_gemm<T> (x_batched, w_batched, y_batched);
			} else if (flag == 2) {
				op_sdmm<T> (x_batched, w_batched, y_batched);	
			}

      for (int i = 0; i < y_batched.size(); i++) {
			  op_add(y_batched[i], b_batched[i], y_batched[i]);
      }
    }
  }
};
