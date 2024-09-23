from setuptools import setup, find_packages

setup(
    license="MIT",
    name="ml_feature_selector",
    version="0.1.2",  # Versioning follows the semantic versioning system
    author="Sangam Man Buddhacharya",
    author_email="sangambuddhacharya@gmail.com",
    description="This package provides code for optimal feature selection using forward and backward wrapper-based methods. It also generates an Excel report that captures each step and result of the feature selection process, offering clear insights and explanations of the selected features.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sanbuddhacharyas/feature_selection",  # Optional
    packages=find_packages(),  # Automatically find your package folders
    install_requires=[ # List any dependencies your package needs, e.g.:
        "scikit-learn>=1.5.1",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "openpyxl>=3.1.2"
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # Minimum Python version
)