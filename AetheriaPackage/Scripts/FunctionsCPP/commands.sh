 sudo yum group install 'Development Tools' -y
 cythonize -a -i -3 MathFunctions.pyx
 python setup.py build_ext --inplace