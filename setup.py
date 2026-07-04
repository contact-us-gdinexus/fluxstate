from setuptools import setup, find_packages

setup(
    name="fluxstate-security",
    version="1.0.1",
    description="Privacy-Preserving Contextual Edge Video Analytics SDK",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="FluxState Security Architecture",
    packages=find_packages(),
    package_data={"fluxstate_security": ["*.json"]},
    include_package_data=True,
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "mediapipe>=0.10.0",
        "ultralytics>=8.0.0",
        "pytesseract>=0.3.10",
        "pyaudio>=0.2.14"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
