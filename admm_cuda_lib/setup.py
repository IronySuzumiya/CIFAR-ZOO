from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='admm_cuda_lib',
    ext_modules=[
        CUDAExtension('admm_cuda_lib', [
            'admm_cuda_lib.cpp',
            'admm_cuda_lib_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })