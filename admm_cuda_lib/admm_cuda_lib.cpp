#include <torch/extension.h>

// CUDA interface
void struct_norm_cuda(
    torch::Tensor weights,
    torch::Tensor norms,
    int grid_w,
    int grid_h);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++ interface
void struct_norm(
    torch::Tensor weights,
    torch::Tensor norms,
    int grid_w,
    int grid_h)
{
    CHECK_INPUT(weights);
    CHECK_INPUT(norms);
    struct_norm_cuda(weights, norms, grid_w, grid_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("struct_norm", &struct_norm, "in-place struct_norm");
}
