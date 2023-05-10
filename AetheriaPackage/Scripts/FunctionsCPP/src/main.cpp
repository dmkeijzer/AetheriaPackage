#include <iostream>
#include "Mechanics.hpp"

using namespace std;

int main (int nargs, char** args ) {
    //Polynomial squared(1, 1);
    Array coefs, roots, exps;
    for (int i=0; i<5; i++) {
        coefs.push(9-5*i);
        roots.push(5*i);
        exps.push(i);
    }
    Polynomial powers(1, 1, "y");
    powers.load(coefs);
    powers.print();
    powers.derivative().print();
    cout << powers.derivative().str() << endl;
    cout << endl << endl;
    Stepfunction My;
    My.load(coefs, roots, exps);
    My.print();
    (My+5).print();
    My.integral(-7).print();
    (-My).print();
    return EXIT_SUCCESS;
}