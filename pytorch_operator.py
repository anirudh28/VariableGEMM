import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from src/ops/op_mm.cuh import batched_gemm
from pycuda.compiler import SourceModule

mod = SourceModule(batched_gemm)

batched_sgemm = mod.get_function("batched_sgemm")

class BatchedMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3-dimensional tensors (batched matrices)"
        batch_size, m, k1 = A.size()
        _, k2, n = B.size()
        assert k1 == k2, "Inner dimensions must match for matrix multiplication"

        C = torch.zeros(batch_size, m, n).cuda()


        A_flat = A.contiguous().view(-1)
        B_flat = B.contiguous().view(-1)
        C_flat = C.contiguous().view(-1)

        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        batched_sgemm(cuda.In(A_flat), cuda.In(B_flat), cuda.Out(C_flat), np.int32(m), np.int32(n), np.int32(k1), np.int32(batch_size), block=(block_size, 1, 1), grid=(grid_size, 1))

        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.transpose(1, 2))

        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.transpose(1, 2), grad_output)

om_mm_cuda = load(name='om_mm_cuda', sources=['om_mm.cu'], verbose=True)

class BatchedGEMMFunction(Function):
    @staticmethod
    def forward(ctx, A_batched, B_batched):
        A_batched_cuda = [tensor.contiguous().float().cuda() for tensor in A_batched]
        B_batched_cuda = [tensor.contiguous().float().cuda() for tensor in B_batched]
        C_batched_cuda = [torch.zeros_like(A).cuda() for A in A_batched_cuda]

        om_mm_cuda.batched_gemm(A_batched_cuda, B_batched_cuda, C_batched_cuda)

        ctx.save_for_backward(*A_batched_cuda, *B_batched_cuda)

        C_batched = [tensor.cpu() for tensor in C_batched_cuda]

        return C_batched

    @staticmethod
    def backward(ctx, grad_output):
        *A_batched_cuda, *B_batched_cuda = ctx.saved_tensors

        grad_A_batched_cuda = [torch.zeros_like(A).cuda() for A in A_batched_cuda]
        grad_B_batched_cuda = [torch.zeros_like(B).cuda() for B in B_batched_cuda]

        grad_output_cuda = [tensor.contiguous().float().cuda() for tensor in grad_output]

        om_mm_cuda.batched_gemm(grad_output_cuda, B_batched_cuda, grad_A_batched_cuda)
        om_mm_cuda.batched_gemm(A_batched_cuda, grad_output_cuda, grad_B_batched_cuda)

        grad_A_batched = [tensor.cpu() for tensor in grad_A_batched_cuda]
        grad_B_batched = [tensor.cpu() for tensor in grad_B_batched_cuda]

        return tuple(grad_A_batched), tuple(grad_B_batched)
        

class BatchedGEMMModule(torch.nn.Module):
    def __init__(self):
        super(BatchedGEMMModule, self).__init__()

    def forward(self, A_batched, B_batched):
        return BatchedGEMMFunction.apply(A_batched, B_batched)


        return grad_A, grad_B
