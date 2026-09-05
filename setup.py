from setuptools import setup, Extension
import numpy

setup(
    name="f1sim_fastcore",
    ext_modules=[
        Extension(
            "f1sim._fastcore",
            sources=["f1sim/_fastcore.cpp"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-march=native", "-ffast-math", "-funroll-loops", "-std=c++17"],
            extra_link_args=["-static-libgcc", "-static-libstdc++"],
        )
    ],
)
