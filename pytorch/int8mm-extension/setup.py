from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int8mm',
    ext_modules=[
        CUDAExtension('int8mm_cuda',
                      ['int8mm.cpp',
                       'int8mm_kernel.cu',],
                      library_dirs=['/usr/lib/x86_64-linux-gnu'],
                      include_dirs=['/media/fa/Shared_Files/Fraunhofer/niti/pytorch/Common'],
                      libraries=['cublas'])
    ],
    #used for letting the cuda and cpp code being debuggable by vsCode (or any other debugger)
    extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
    cmdclass={
        'build_ext': BuildExtension
    })
