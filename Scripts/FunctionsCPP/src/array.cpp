#include <iostream>

using namespace std;

double pow (double base, double exp) {
    double prod = 1;
    for (int i = 0; i < int(exp); i++) {
        prod *= base;
    };
    return prod;
};

class Array {
    double *list = new double[0];
    public:
        int size = 0;
        void push(double item) {
            double *newlist = new double[size + 1];
            for (int i=0; i < size; i++) {
                newlist[i] = list[i];
            };
            delete [] list;
            newlist[size++] = item;
            list = newlist;
        };
    
        double& operator[] (int index) {
            return list[index];
        };

        Array slice (int s, int e) {
            Array newarr;
            for (int i = s; i < e; i++) {
                newarr.push(list[i]);
            }
            return newarr;
        }

        double pop(int index) {
            double *newlist = new double[--size];
            double val;
            for (int i=0; i < size + 1; i++) {
                if (i != index) {
                    newlist[i] = list[i];
                } else {
                    val = list[i];
                    break;
                }
            } 
            for (int j=index + 1; j < size + 1; j++) {
                newlist[j-1] = list[j];
            }
            delete [] list;
            list = newlist;
            return val;
        }

        int remove (double value) {
            int index;
            for (int i=0; i < size + 1; i++) {
                if (list[i] == value) {
                    index = i;
                    this->pop(i);
                    break;
                }
            }
            return index;
        }

        string join(string del) {
            string joined, elem;
            for (int i=0; i<size-1; i++) {
                elem = to_string(list[i]);
                elem.append(del);
                joined.append(elem);
            }
            joined.append(to_string(list[size-1]));
            return joined;
        }

        Array map (double (*f) (double) ) {
            Array retarr;
            for (int i=0; i<size; i++) {
                retarr.push(f(list[i]));
            }
            return retarr;
        }

        Array filter (bool (*f) (double, int) ) {
            Array retarr;
            for (int i=0; i<size; i++) {
                if (f(list[i], i) == true) {
                    retarr.push(list[i]);
                }
            }
            return retarr;
        }

        double reduce (double (*f) (double, double, int)) {
            double agg = 0;
            for (int i=0; i<size; i++) {
                agg = f(list[i], agg, i);
            }
            return agg;
        }
         void print( ) {
            cout << "{" << list[0];
            for (int i=1; i<size; i++) {
                cout << ", " << list[i];
            } cout << "}\n";
        }
        void forEach(void (*f) (double)) {
            for (int i=0; i<size; i++) {
                f(list[i]);
            };
        };
        double max () {
            return this->reduce([](double li, double agg, int i) {return li > agg ? li : agg;});
        };
        double min () {
            return this->reduce([](double li, double agg, int i) {return li < agg ? li : agg;});
        };
        int in ( double item ) {
            for (int i = 0; i < this->size; i++) {
                if (item == list[i]) {
                    return i;
                };
            };
            return -1;
        };
};