"""
Build script for moe-infer-mlx Cython extension.

Compiles the C wrapper (src/flashmoe_c.m, Objective-C) together with
the Cython bridge (moe_infer_mlx/_core.pyx) into a single shared library.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "moe_infer_mlx._core",
    sources=[
        "moe_infer_mlx/_core.pyx",
        "src/flashmoe_c.m",
    ],
    include_dirs=["src", np.get_include()],
    extra_compile_args=[
        "-O2",
        "-Wall",
        "-fobjc-arc",
        "-DACCELERATE_NEW_LAPACK",
    ],
    extra_link_args=[
        "-lpthread",
        "-lcompression",
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "Accelerate",
    ],
)

setup(
    name="moe-infer-mlx",
    version="0.1.0",
    python_requires=">=3.10",
    ext_modules=cythonize(
        [ext],
        language_level="3",
    ),
    packages=["moe_infer_mlx"],
)
