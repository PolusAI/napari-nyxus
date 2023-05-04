
import versioneer

from setuptools import setup


setup(
    name="napari-nyxus",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)