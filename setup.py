import os

from setuptools import setup

if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]


SKIP_CUDA_BUILD = os.environ.get("SILEX_SKIP_CUDA_BUILD", "").lower() in {"1", "true", "yes"}


def _cuda_ext_modules():
    if SKIP_CUDA_BUILD:
        return []

    from torch.utils.cpp_extension import CUDAExtension

    if os.name == "nt":
        cxx_flags = ["/O2", "/Zc:preprocessor"]
        nvcc_flags = ["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"]
    else:
        cxx_flags = ["-O2"]
        nvcc_flags = ["-O3", "--use_fast_math"]

    return [
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
    ]


def _cmdclass():
    if SKIP_CUDA_BUILD:
        return {}
    from torch.utils.cpp_extension import BuildExtension

    return {"build_ext": BuildExtension}


setup(
    name="silexcode",
    version="0.1.0",
    packages=["silexcode"],
    ext_modules=_cuda_ext_modules(),
    cmdclass=_cmdclass(),
)
