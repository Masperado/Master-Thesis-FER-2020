import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyb_manipulator",
    version="0.0.1",
    author="Filip Maric",
    author_email="filip.amric@robotics.utias.utoronto.ca",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/utiasSTARS/pyb-manipulator",
    packages=setuptools.find_packages(),
)