"""Setup and Install Script."""


from setuptools import setup


def get_readme():
    """Load README.rst for display on PyPI."""
    with open('README.md') as fhandle:
        return fhandle.read()


setup(
    name="SGAD",
    version="0.0.1b1",
    description="SGAD Package",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    url="https://github.com/ComputeSuda/SGAD",
    license="GNU (Version 3)",
    author="ComputeSuda",
    package_dir={"SGAD": "src"},
    packages=["src"],
    install_requires=["numpy>=1.18.5", "scipy>=1.5.0"],
)
