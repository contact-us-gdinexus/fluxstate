from setuptools import setup, find_packages

setup(
    name="fluxstate-security",
    version="1.1.0",
    description="Privacy-Preserving Contextual Edge Video Analytics SDK",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="GDI Nexus Software Solutions LLP",
    author_email="admin@gdinexus.in",
    url="https://github.com/contact-us-gdinexus/fluxstate",
    license="MIT",
    packages=find_packages(),
    package_data={"fluxstate_security": ["*.json"]},
    include_package_data=True,
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "mediapipe>=0.10.0",
        "ultralytics>=8.0.0",
        "pytesseract>=0.3.10",
        "pyaudio>=0.2.14",
        "PyJWT>=2.0.0"
    ],
    extras_require={
        "apple": ["mlx>=0.16.0", "mlx-vlm>=0.0.9"]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
