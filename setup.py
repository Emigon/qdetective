from setuptools import setup, find_packages

setup(name='qdetective',
      version='0.0.1',
      description='Qdetective: Analysis tools for superconducting resonators',
      url='https://github.com/Emigon/qdetective',
      author='Daniel Parker',
      author_email='danielparker@live.com.au',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.16.3',
          'pandas>=0.24.0'
          'matplotlib',
        ])
