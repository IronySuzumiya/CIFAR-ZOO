from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='admm_cuda_lib',
    ext_modules=[
        CUDAExtension('admm_cuda_lib', [
            'struct_norm.cpp',
            'struct_norm_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })