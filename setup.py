import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="olivia",
    version="1.0.0",
    author="Dongwei Li, Michael X. Wang, Lauren B. Stadler, Todd J. Treangen",
    description="Olivia: learned sequence representations for multiplex tiled amplicon primer design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/treangenlab/Olivia",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    package_dir={"": "src"},
    packages=["olivia", "olivia_encoder"],
    scripts=["olivia"],
    python_requires=">=3.10",
)
