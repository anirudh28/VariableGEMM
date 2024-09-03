#pragma once

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>


#include "utils/cublas_utils.h"
#include "utils/tensor.cuh"

template <typename T>
__global__ void vjoin_cpymat_kernel(const Tensor<T> A, T *dest) {
	int row = blockIdx.x;
	int col = threadIdx.x;

	dest[row * A.w + col] = Index(A, row, col);
}

template <typename T>
__global__ void hjoin_cpymat_kernel(const Tensor<T> A, T *dest, int row_offset, int row_size) {
	int row = blockIdx.x;
	int col = threadIdx.x;

	dest[row_size * row + row_offset + col] = Index(A, row, col);
}

template <typename T>
__global__ void extract_mat_kernel(Tensor<T> A, T *src) {
	int row = blockIdx.x;
	int col = threadIdx.x;
	// printf("row = %d \t col = %d \n", row, col);
	return;
	Index(A, row, col) = src[row * A.w + col];
}

template <typename T>
__global__ void vjoin_matrices_kernel(const Tensor<T> *A_arr, T *dest, int dest_numel) {
	int mat_idx = threadIdx.x;
	int start_idx = 0;
	for (int i = 0; i < mat_idx; i++) {
		start_idx = start_idx + A_arr[i].h * A_arr[i].w;
	}
	for (int i = 0; i < A_arr[mat_idx].h; i++) {
		for (int j = 0; j < A_arr[mat_idx].w; j++) {
			dest[start_idx] = Index(A_arr[mat_idx], i, j);
			start_idx++;
		}
	}
}

template <typename T>
__global__ void hjoin_matrices_kernel(const Tensor<T> *A_arr, T *dest, int A_arr_numel, const int n_tot) {
	int row = threadIdx.x;
	int idx = n_tot * row;

	for (int i = 0; i < A_arr_numel; i++) {
		for (int j = 0; j < A_arr[i].w; j++) {
			dest[idx] = Index(A_arr[i], row, j);
			idx++;
		}
	}
}

template <typename T>
__global__ void extract_tensors_kernel(const T *dC_values, Tensor<T> *C) {
	int mat_idx = threadIdx.x;
	int start_idx = 0;
	for (int i = 0; i < mat_idx; i++) {
		start_idx = start_idx + C[mat_idx].h * C[mat_idx].w;
	}

	for (int i = 0; i < C[mat_idx].h; i++) {
		for (int j = 0; j < C[mat_idx].w; j++) {
			Index(C[mat_idx], i, j) = dC_values[start_idx + i * C[mat_idx].w + j];
		}
	}
}

template <typename T>
void op_sdmm (const std::vector<Tensor<T>> &A, const std::vector<Tensor<T>> &B, std::vector<Tensor<T>> &C) {
	int gemm_count = A.size();
	for (int idx = 0; idx < gemm_count; idx++) {
		assert(A[idx].on_device && B[idx].on_device && C[idx].on_device);
		assert(A[idx].w == B[idx].h && A[idx].h == C[idx].h && B[idx].w == C[idx].w);
	}	

	int m_tot = 0;
	int k = 0;
	int n_tot = 0;
	int C_nnz = 0;
	for (int idx = 0; idx < gemm_count; idx++) {
		m_tot += A[idx].h;
		n_tot += B[idx].w;
		C_nnz += A[idx].h * B[idx].w;
	}
	k = A[0].w;
	
	int *hC_offsets = new int[m_tot + 1];
	int *hC_columns = new int[C_nnz];
	T *hC_values 		= new T[C_nnz];

	hC_offsets[0] = 0;
	int idx = 1;
	for (int i = 0; i < gemm_count; i++) {
		for (int j = 0 ; j < A[i].h; j++) {
			hC_offsets[idx] = hC_offsets[idx - 1] + B[i].w;
			idx++;
		}
	}
	idx = 0;
	int offset = 0;
	for (int i = 0; i < gemm_count; i++) {
		for (int j = 0; j < A[i].h; j++) {
			for (int l = 0; l < B[i].w; l++) {
				hC_columns[idx] = offset + l;
			}
		}
		offset += B[i].w;
	}
	memset(hC_values, 0, C_nnz);
	for (int i = 0; i < C_nnz; i++) hC_values[i] = 1;
	
	int *dC_offsets, *dC_columns;
	T *dA_dense, *dB_dense, *dC_dense, *dC_values;
	int lda = k, ldb = n_tot, ldc = n_tot;
		
	CUDA_CHECK( cudaMalloc((void **) &dA_dense, m_tot * k * sizeof(T)) );
	CUDA_CHECK( cudaMalloc((void **) &dB_dense, k * n_tot * sizeof(T)) );
	CUDA_CHECK( cudaMalloc((void **) &dC_dense, m_tot * n_tot * sizeof(T)) );
	CUDA_CHECK( cudaMalloc((void **) &dC_offsets, (m_tot + 1) * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void **) &dC_columns, C_nnz * sizeof(int)) );
	CUDA_CHECK( cudaMalloc((void **) &dC_values, C_nnz * sizeof(T)) );

	CUDA_CHECK( cudaDeviceSynchronize() );	
	std::cout << "gemm_count = " << gemm_count << std::endl;
	// join A tensors verticaly into dA_dense
//	vjoin_matrices_kernel<T><<<1, gemm_count>>>(A.data(), dA_dense, m_tot * k);
	int start_idx = 0;
	for (int i = 0; i < gemm_count; i++) {
		vjoin_cpymat_kernel<T><<<A[i].h, A[i].w>>>(A[i], (T *)(dA_dense + start_idx * sizeof(T)));
		start_idx += A[i].h * A[i].w;
	}
	
	CUDA_CHECK( cudaDeviceSynchronize() );
	
	// join B tensor horizontally into dB_dense
//	hjoin_matrices_kernel<T><<<1, k>>>(B.data(), dB_dense, B.size(), n_tot);
	int row_offset = 0;
	for (int i = 0; i < gemm_count; i++) {
		hjoin_cpymat_kernel<T><<<B[i].h, B[i].w>>>(B[i], dB_dense, row_offset, n_tot);
		row_offset += B[i].w;
	}
	// set dC_offsets and dC_columns	

	std::printf("C_nnz = %d \t m_tot = %d \t n_tot = %d \n", C_nnz, m_tot, n_tot);
/*
	cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
	T *d_C = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(T) * m_tot * n_tot));
	cublasHandle_t cublasH = NULL;
	CUBLAS_CHECK(cublasCreate(&cublasH));
	T alpha = 1.0f, beta = 0.0f;
	CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, m_tot, n_tot, k, &alpha, dA_dense, m_tot, dB_dense, k, &beta, d_C, m_tot));
*/
	T *alpha = new T; *alpha =  1.0;
	T *beta = new T; *beta =  1.0;
	CUDA_CHECK( cudaDeviceSynchronize() );

	CUDA_CHECK( cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(T), cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemcpy(dC_offsets, hC_offsets, (m_tot + 1) * sizeof(int), cudaMemcpyHostToDevice) );

	CUDA_CHECK( cudaDeviceSynchronize() );

	// cuSPARSE APIs
	cusparseHandle_t 			handle = NULL;
	cusparseDnMatDescr_t 	matA, matB, matC_dense;	
	cusparseSpMatDescr_t 	matC;
	void*									dBuffer = NULL;
	size_t								bufferSize = 0;
// 	T alpha = 1.0f, beta = 0.0f;

	CHECK_CUSPARSE( cusparseCreate(&handle) );
	CHECK_CUSPARSE( cusparseCreateDnMat(&matA, m_tot, k, lda, dA_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW) );
	CHECK_CUSPARSE( cusparseCreateDnMat(&matB, k, n_tot, ldb, dB_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW) );
	CHECK_CUSPARSE( cusparseCreateDnMat(&matC_dense, m_tot, n_tot, ldc, dC_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW) );
	CHECK_CUSPARSE( cusparseCreateCsr(&matC, m_tot, n_tot, C_nnz, dC_offsets, dC_columns, dC_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );	
	CUDA_CHECK( cudaDeviceSynchronize() );

	CHECK_CUSPARSE( cusparseSparseToDense_bufferSize(
																				handle, matC, matC_dense,
                                        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                        &bufferSize) );
	CUDA_CHECK( cudaMalloc(&dBuffer, bufferSize) );
	std::cout << " buffer size = " << bufferSize << std::endl; 
	CHECK_CUSPARSE( cusparseSparseToDense(handle, matC, matC_dense,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          dBuffer) );
	CUDA_CHECK( cudaDeviceSynchronize() );	
	
	CHECK_CUSPARSE( cusparseSDDMM_bufferSize(
															handle,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																alpha, matA, matB, beta, matC, CUDA_R_32F,
																CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) );

	CUDA_CHECK( cudaMalloc(&dBuffer, bufferSize) );

	CUDA_CHECK( cudaDeviceSynchronize() );
	// execute preprocess
	CHECK_CUSPARSE( cusparseSDDMM_preprocess(
																handle,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																alpha, matA, matB, beta, matC, CUDA_R_32F,
																CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) );
	
//	CUDA_CHECK( cudaDeviceSynchronize() );
	
	// execute SpMM
	CHECK_CUSPARSE( cusparseSDDMM(handle,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																CUSPARSE_OPERATION_NON_TRANSPOSE,
																&alpha, matA, matB, &beta, matC, CUDA_R_32F,
																CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) );

	CUDA_CHECK( cudaDeviceSynchronize() );
/*	
	CHECK_CUSPARSE( cusparseDestroyDnMat(matA) );
	CHECK_CUSPARSE( cusparseDestroyDnMat(matB) );
	CHECK_CUSPARSE( cusparseDestroySpMat(matC) );
	CHECK_CUSPARSE( cusparseDestroy(handle) );
*/
 	CUDA_CHECK( cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float), cudaMemcpyDeviceToHost) );
	for (int i = 0; i < C_nnz; i++) {
		std::cout << hC_values[idx] << ' ';
	} std::cout << std::endl;
	offset = 0;
	std::cout << "C_size = " << C.size() << std::endl;	
	for (int i = 0; i < gemm_count; i++) {
		std::cout << C[i].h << ' ' << C[i].w << ' ' << dC_values << ' ' << std::endl;
		extract_mat_kernel<T><<<C[i].h, C[i].w>>>(C[i], (T *)(dC_values + offset * sizeof(T)));
		offset += A[i].h * B[i].w;
		CUDA_CHECK( cudaDeviceSynchronize() );
		printf("copied C_mat %d\n", i);
	}
	CUDA_CHECK( cudaDeviceSynchronize() );	
//`extract_tensors_kernel<T><<<1, gemm_count>>>(dC_values, C.data());

	CHECK_CUSPARSE( cusparseDestroyDnMat(matA) );
  CHECK_CUSPARSE( cusparseDestroyDnMat(matB) );
  CHECK_CUSPARSE( cusparseDestroySpMat(matC) );
  CHECK_CUSPARSE( cusparseDestroy(handle) );

	// free device memory
	CUDA_CHECK( cudaFree(dBuffer) );
  CUDA_CHECK( cudaFree(dA_dense) );
  CUDA_CHECK( cudaFree(dB_dense) );
  CUDA_CHECK( cudaFree(dC_offsets) );
  CUDA_CHECK( cudaFree(dC_columns) );
  CUDA_CHECK( cudaFree(dC_values) );
	
	// free host memory
	delete[] hC_offsets;
	delete[] hC_columns;
	delete[] hC_values;

}
