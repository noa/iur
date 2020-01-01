from setuptools import setup
from setuptools import find_packages

setup(name='aid',
      version='0.1',
      description='TensorFlow-based Author Identification',
      url='https://github.com/noa/iur',
      author='Nicholas Andrews and Marcus Bishop',
      author_email='noa@jhu.edu',
      license='Apache 2.0',
      packages=find_packages(),
      zip_safe=False)
