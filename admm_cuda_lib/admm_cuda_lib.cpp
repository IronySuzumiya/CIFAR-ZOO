#include <torch/extension.h>

// CUDA interface
void struct_norm_nxn2nx1_cuda(
    torch::Tensor weights,
    torch::Tensor norms);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++ interface
void struct_norm_nxn2nx1(
    torch::Tensor weights,
    torch::Tensor norms)
{
    CHECK_INPUT(weights);
    CHECK_INPUT(norms);
    struct_norm_nxn2nx1_cuda(weights, norms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("struct_norm_nxn2nx1", &struct_norm_nxn2nx1, "in-place struct_norm_nxn2nx1");
}
