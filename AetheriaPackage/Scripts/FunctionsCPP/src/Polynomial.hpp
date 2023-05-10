#include <sstream>
#include <iostream>
#include "array.cpp"

using namespace std;

struct Polynomial  {
    double f, c;
    Array arr;
    string s;
    Polynomial (double freq = 1, double coef = 1, string symbol = "x") {
        f = freq; c = coef; s = symbol;
    };
    double operator()(double x) {
        double sum = 0;
        for (int i=0; i<arr.size; i++) {
            sum += arr[i] * pow(x*f, i);
        };
        return c * sum;
    };
    void push(double term) {
        arr.push(term);
    };
    void load(Array aray) {
        for (int i = 0; i < aray.size; i++) {
            arr.push(aray[i]);
        };
    };
    void print() {
        cout << this->str() << "\n";
    };

    string str ( ) {
        stringstream out;
        if (c != 1) {
            out << c << "*[";
        } else {
            out << (c == 1 ? "" : "[");
        };
        if (arr[0] != 0) {
            out << arr[0] << " + ";
        };
        for (int i=1; i<arr.size-1; i++) {
            if (arr[i] != 0) {
                out << arr[i] << "*(" << f << "*" << s << ")^" << i << " + ";
            };
        };
        out << arr[arr.size - 1] << "*(" << f << "*" << s << ")^" << arr.size - 1;
        out << ( c == 1 ? "" : "]") << "\n";
        return out.str();
    };

    Polynomial derivative() {
        Polynomial newpol(this->f, (this->c)*(this->f), this->s);
        for (int i=1; i <arr.size; i++) {
            newpol.push((i * arr[i]));
        };
        return newpol;
    };

    Polynomial integral(double C = 0) {
        Polynomial newpol(this->f, (this->c)*(this->f), this->s);
        newpol.push(C * (this->f) / (this->c));
        for (int i = 0; i < arr.size; i++) {
            newpol.push(arr[i] / (i + 1));
        };
        return newpol;
    };
    double dintegral(double a, double b) {
        return (this->integral(0))(b) - (this->integral(0))(a);
    };
    Polynomial operator+(double p) {
        Polynomial newpol(this->f, (this->c), this->s);
        newpol.push(p * (this->f) / (this->c)); 
        newpol.load(arr.slice(1, arr.size));
        return newpol;
    };

    Polynomial operator*(double p) {
        Polynomial newpol(this->f, p * (this->c), this->s);
        newpol.load(arr);
        return newpol;
    };

    Polynomial operator/(double p) {
        Polynomial newpol(this->f, (this->c) / p, this->s);
        newpol.load(arr);
        return newpol;
    };
    Polynomial operator-() {
        Polynomial newpol(this->f, -(this->c), this->s);
        newpol.load(arr);
        return newpol;
    };

    double optimize ( double a, double b ) {
            const double gold = 1.61803398875;
            double beta = 1 / gold;
            double alpha = 1 - beta;
            double xl, xr;
            while ( abs(b - a) > 1e-8 ) {
                xl = a + alpha*(b-a);
                xr = a + beta*(b-a);
                if ( (*this)(xl) < (*this)(xr) ) {
                    b = xr;
                } else {
                    a = xl;
                }
            };
            return (a + b) / 2;
        };
        double max (double a, double b) {
            Polynomial negative = -(*this);
            return (*this)(negative.optimize(a, b));
        };

        double min (double a, double b) {
            return (*this)(this->optimize(a, b));
        };
};

Polynomial operator+(double a, Polynomial poly) {
    return poly + a;
};

Polynomial operator*(double a, Polynomial poly) {
    return poly * a;
};

Polynomial operator-(Polynomial poly, double p) {
    return poly + (-p);
};

Polynomial operator-(double p, Polynomial poly) {
    return p + (-poly);
};

Polynomial operator+(Polynomial poly1, Polynomial poly2) {
    int max = poly2.arr.size < poly1.arr.size ? poly1.arr.size : poly2.arr.size;
    Polynomial sum(1, 1, poly1.s); double val1, val2;
    for (int i=0; i<max; i++) {
        val1 = i > poly1.arr.size ? 0 : poly1.arr[i]*poly1.c*pow(poly1.f, i);
        val2 = i > poly2.arr.size ? 0 : poly2.arr[i]*poly2.c*pow(poly2.f, i);
        sum.push(val1 + val2);
    };
    return sum;
};
