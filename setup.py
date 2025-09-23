import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the current version of the lib (avoids double sourcing in code and pyproject.toml)."""
    path = CWD / "engiopt" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="engiopt",
    version=get_version(),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown"
)
