import os

if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if os.name == "nt":
    cxx_flags = ["/O2", "/Zc:preprocessor"]
    nvcc_flags = ["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"]
else:
    cxx_flags = ["-O2"]
    nvcc_flags = ["-O3", "--use_fast_math"]


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
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
