#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
template <typename scalar_t>
__global__ void struct_norm_nxn2nx1_cuda_kernel(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weight,
                          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> norm,
                          int Wx,
                          int Wy)
{
    int i = blockIdx.x * 32 + threadIdx.x;
    if (i < Wx) {
      scalar_t sum = 0;
      for (int j = 0; j < Wy; ++j) {
        sum += weight[i][j] * weight[i][j];
      }
      norm[i] = sqrt(sum);
    }
}

void struct_norm_nxn2nx1_cuda(
    torch::Tensor weights,
    torch::Tensor out_norm)
{
    const auto WeightsSizeX = weights.size(0);
    const auto WeightsSizeY = weights.size(1);
    dim3 threadDim(32, 1);
    dim3 blockDim(((WeightsSizeX - 1) / 32 + 1), 1);
    
    AT_DISPATCH_FLOATING_TYPES(weights.type(), "struct_norm_nxn2nx1_cuda", ([&] {
      struct_norm_nxn2nx1_cuda_kernel<scalar_t><<<blockDim, threadDim>>>(
          weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          out_norm.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          WeightsSizeX,
          WeightsSizeY);
    }));
}
