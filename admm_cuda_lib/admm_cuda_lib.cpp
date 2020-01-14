#include <torch/extension.h>

// CUDA interface
void struct_norm_cuda(
    torch::Tensor weights,
    torch::Tensor norms,
    int ou_w,
    int ou_h);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++ interface
void struct_norm(
    torch::Tensor weights,
    torch::Tensor norms,
    int ou_w,
    int ou_h)
{
    CHECK_INPUT(weights);
    CHECK_INPUT(norms);
    struct_norm_cuda(weights, norms, ou_w, ou_h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("struct_norm", &struct_norm, "in-place struct_norm");
}
