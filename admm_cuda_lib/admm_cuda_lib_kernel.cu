#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
template <typename scalar_t>
__global__ void struct_norm_cuda_kernel(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weight,
                          torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> norm,
                          int grid_w, int grid_h,
                          int Wx,
                          int Wy,
                          int Nx,
                          int Ny)
{
    int startX = (blockIdx.x * 8 + threadIdx.x) * grid_w;
    int startY = (blockIdx.y * 8 + threadIdx.y) * grid_h;
    if (startX < Wx && startY < Wy) {
      scalar_t sum = 0;
      for (int i = 0; i < grid_w && startX + i < Wx; i++) {
        for (int j = 0; j < grid_h && startY + j < Wy; j++) {
          sum += weight[startX + i][startY + j] * weight[startX + i][startY + j];
        }
      }
      norm[blockIdx.x * 8 + threadIdx.x][blockIdx.y * 8 + threadIdx.y] = sqrt(sum);
    }
}

void struct_norm_cuda(
    torch::Tensor weights,
    torch::Tensor out_norm,
    int grid_w,
    int grid_h)
{
    const auto WeightsSizeX = weights.size(0);
    const auto WeightsSizeY = weights.size(1);
    const auto NormSizeX = out_norm.size(0);
    const auto NormSizeY = out_norm.size(1);
    dim3 threadDim(8, 8);
    dim3 blockDim(((NormSizeX - 1) / 8 + 1), ((NormSizeY - 1) / 8 + 1));
    
    AT_DISPATCH_FLOATING_TYPES(weights.type(), "struct_norm_cuda", ([&] {
      struct_norm_cuda_kernel<scalar_t><<<blockDim, threadDim>>>(
          weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          out_norm.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          grid_w,
          grid_h,
          WeightsSizeX,
          WeightsSizeY,
          NormSizeX,
          NormSizeY);
    }));
}