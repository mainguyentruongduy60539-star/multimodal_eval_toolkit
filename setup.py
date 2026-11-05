from setuptools import setup, find_packages

setup(
    name="multimodal_eval_toolkit",
    version="1.0.0",
    description="Khung đánh giá đa tiêu chí cho mô hình hợp nhất y sinh dựa trên Fusion Gain, IMR và UQ",
    author="Nhom 04 - Pham Dinh Hung & Mai Nguyen Truong Duy",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0"
    ],
)