from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

print(find_packages(where="src"))

setup(
    name="idoven_data_scientist",
    version="0.0.0",
    description="Analysis suite for the test dataset of Idoven",
    packages=["idoven_data_scientist"],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "analysis=idoven_data_scientist.analysis:analyse_single_record",
        ],
    },
)
