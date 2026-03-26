from setuptools import setup, find_packages

setup(
    name="fairness_constraint",
    version="1.0.0",
    description="Fair supervised learning through constraints on smooth nonconvex unfairness-measure surrogates",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "matplotlib>=3.0.0",
        "torch>=1.7.0",
        "scikit-learn>=0.24.0",
        "folktables>=0.0.11",
        "psutil",
    ],
    entry_points={
        "console_scripts": [
            "fairness-train=main_train:main",
            "fairness-plot=main_plot:main",
        ],
    },
)
