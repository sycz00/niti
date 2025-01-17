from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cutlassconv',
    ext_modules=[
        CUDAExtension(name='cutlassconv_cuda',
                      sources=['cutlassconv.cpp',
                       'cutlassconv_kernel.cu',],
                      include_dirs=['/media/fa/Shared_Files/Fraunhofer/niti/pytorch/cutlass-extension/include','/media/fa/Shared_Files/Fraunhofer/niti/pytorch/cutlass-extension/util/include'])],
                      #used for letting the cuda and cpp code being debuggable by vsCode (or any other debugger) remove it for deployment ?
                      extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
    cmdclass={
        'build_ext': BuildExtension
    })
