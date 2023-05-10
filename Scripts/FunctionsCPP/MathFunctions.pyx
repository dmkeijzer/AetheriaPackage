# distutils: language = c++
from libcpp.string cimport string

cdef extern from "src/Mechanics.hpp":
    cdef cppclass Stepfunction:
        Stepfunction(string) except +
        Stepfunction() except +
        void push(double coef, double root, double exp)
        void print()
        double& get "operator()"(double)
        Stepfunction derivative()
        Stepfunction integral(double)
        Stepfunction operator+(Stepfunction)
        Stepfunction operator-()
        Stepfunction& mul "operator*"(double)
        Stepfunction& div "operator/"(double)
        double optimize(double, double)
        double max(double, double)
        double min(double, double)
        string str()

cdef extern from "src/Mechanics.hpp":
    cdef cppclass Polynomial:
        Polynomial(double, double, string) except +
        Polynomial(double, double) except +
        Polynomial() except +
        void push(double)
        void print()
        double& get "operator()"(double)
        Polynomial derivative()
        Polynomial integral(double)
        Polynomial operator+(Polynomial)
        Polynomial operator-()
        Polynomial& mul "operator*"(double)
        Polynomial& div "operator/"(double)
        double optimize(double, double)
        double max(double, double)
        double min(double, double)
        string str()


cdef class StepFunction:
    cdef Stepfunction sf
    def __cinit__(self, list arr = [[0]*3], str symbol = "z"):
        self.sf = Stepfunction(symbol.encode())
        for c, r, e in arr:
            self.sf.push(c, r, e)
    def push(self, double coef, double root, double exp):
        self.sf.push(coef, root, exp)
    def print(self):
        self.sf.print()
    def __call__(self, double z):
        return self.sf.get(z)
    def __str__(self):
        return self.sf.str().decode('utf-8')
    __repr__ = __str__

    @staticmethod
    cdef create(Stepfunction sfn):
        fn = StepFunction()
        fn.sf = sfn
        return fn

    def __add__(StepFunction left, StepFunction right):
        return StepFunction.create(left.sf + right.sf)

    __radd__ = __add__

    def derivative(self):
        return StepFunction.create(self.sf.derivative())
    
    def integral(self, double C = 0):
        newsf = StepFunction()
        newsf.sf = self.sf.integral(C)
        return newsf
    def __neg__(self):
        return StepFunction.create(-self.sf)
    def __sub__(StepFunction left, StepFunction right):
        return StepFunction.create(left.sf + (-right.sf))
    def __rsub__(StepFunction left, StepFunction right):
        return StepFunction.create(-left.sf + right.sf)
    def __mul__(StepFunction a, double b):
        return StepFunction.create(a.sf.mul(b))
    __rmul__ = __mul__

    def max(self, double a, double b):
        return self.sf.max(a, b)
    def min(self, double a, double b):
        return self.sf.min(a, b)

    def __truediv__(StepFunction a, double b):
        return StepFunction.create(a.sf.div(b))

cdef class Poly:
    cdef Polynomial poly
    def __cinit__(self, list arr = [], double freq = 1, double coef = 1, str symbol = "z"):
        self.poly = Polynomial(freq, coef, symbol.encode())
        for t in arr:
            self.poly.push(t)
    def push(self, double term):
        self.poly.push(term)
    def print(self):
        self.poly.print()
    def __call__(self, double z):
        return self.poly.get(z)
    def __str__(self):
        return self.poly.str().decode('utf-8')
    __repr__ = __str__

    @staticmethod
    cdef create(Polynomial polyn):
        fn = Poly()
        fn.poly = polyn
        return fn

    def __add__(Poly left, Poly right):
        return Poly.create(left.poly + right.poly)

    __radd__ = __add__

    def derivative(self):
        return Poly.create(self.poly.derivative())
    
    def integral(self, double C = 0):
        return Poly.create(self.poly.integral(C))
    def __neg__(self):
        return Poly.create(-self.poly)
    def __sub__(Poly left, Poly right):
        return Poly.create(left.poly + (-right.poly))
    def __rsub__(Poly left, Poly right):
        return Poly.create(-left.poly + right.poly)
    def __mul__(Poly a, double b):
        return Poly.create(a.poly.mul(b))
    def max(self, double a, double b):
        return self.poly.max(a, b)
    def min(self, double a, double b):
        return self.poly.min(a, b)

    __rmul__ = __mul__
    __repr__ = __str__

    def __truediv__(Poly a, double b):
        return Poly.create(a.poly.div(b))