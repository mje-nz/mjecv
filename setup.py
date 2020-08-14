import setuptools

about = {}  # type: ignore
with open("mjecv/__about__.py") as f:
    exec(f.read(), about)

with open("Readme.md", "r") as f:
    long_description = f.read()
description = long_description.splitlines()[1].strip("> ")

setuptools.setup(
    name="mjecv",
    version=about["__version__"],
    author="Matthew Edwards",
    author_email="mje-nz@users.noreply.github.com",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mje-nz/mjecv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy>=1.18,<2"],
    extras_requires={
        "opencv": ["opencv-python>=4.3.0,<5", "pickle5>=0.0.11,<0.0.1"],
        "ray": ["ray>=0.8.6,<0.9"],
    },
)
