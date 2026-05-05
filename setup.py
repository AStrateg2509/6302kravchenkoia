"""Минимальный setup.py для регистрации C-расширения; метаданные в pyproject.toml."""

from setuptools import Extension, setup

image_ops = Extension(
    name="metetl.images.c_ext.image_ops",
    sources=["src/metetl/images/c_ext/image_ops.c"],
    optional=True,
)

setup(ext_modules=[image_ops])
