from setuptools import setup, find_packages

setup(
    name="llm-evaluation-template",
    version="1.0.0",
    description="A flexible template for evaluating LLM responses with customizable metrics and MLflow tracking",
    author="AI Evaluation Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mlflow>=3.0",
        "langchain_openai>=0.1.0",
        "pyyaml>=6.0",
        "pandas>=1.5.0",
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "llm-evaluate=evaluation_template:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)