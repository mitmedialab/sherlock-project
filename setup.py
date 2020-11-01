import setuptools

setuptools.setup(
    name="sherlock",
    version="0.0.1",
    author="Madelon Hulsebos",
    author_email="m.hulsebos@uva.nl",
    description="Package for semantic type detection using Sherlock",
    url="https://github.com/mitmedialab/sherlock-project",
    packages=setuptools.find_packages(),
    package_dir={"sherlock": "sherlock"}
)