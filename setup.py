# setup.py (at repo root)
from setuptools import setup
import versioneer

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
