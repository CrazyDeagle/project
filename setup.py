import os

if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="silexcode",
    version="0.1.0",
    packages=["silexcode"],
    ext_modules=[
        CUDAExtension(
            name="silexcode._C",
            sources=[
                "silexcode/cuda/bindings.cpp",
                "silexcode/cuda/tlinear_kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2", "/Zc:preprocessor"] if os.name == "nt" else ["-O2"],
                "nvcc": ["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
