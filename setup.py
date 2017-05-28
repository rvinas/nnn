from setuptools import setup
from setuptools import find_packages
import pip

setup(name='nnn',
      version='0.0.1',
      description='Numpy Neural Network',
      author='Ramon Viñas',
      author_email='rvinast@gmail.com',
      url='https://github.com/rvinas/nnn',
      license='Apache',
      packages=find_packages())

pip.main(['install', 'numpy'])
pip.main(['install', 'matplotlib'])