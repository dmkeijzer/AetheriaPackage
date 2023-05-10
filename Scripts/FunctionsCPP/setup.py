from setuptools import setup
from Cython.Build import cythonize

setup(
  ext_modules = cythonize("MathFunctions.pyx", language="c++"),
  name = 'MathFunctions',
  version = '0.73',
  license='MIT',
  description = 'Programmatically Create and Manipulate Mathematical Functions',
  author = 'Kaizad Wadia',
  author_email = 'kaizad@email.com',
  url = 'https://github.com/chezzoba/',
  keywords = ['Math', 'Numbers', 'Functions', 'Trigonometry', 'Log'],
  install_requires=[],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10'
  ],
)