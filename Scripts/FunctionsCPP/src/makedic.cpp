#include <iostream>

using namespace std;

double pow (double base, double exp) {
    double prod = 1;
    for (int i = 0; i < int(exp); i++) {
        prod *= base;
    };
    return prod;
};

template <typename T>
class Array {
    T *list = new T[0];
    public:
        int size = 0;
        void push(T item) {
            T *newlist = new T[size + 1];
            for (int i=0; i < size; i++) {
                newlist[i] = list[i];
            };
            delete [] list;
            newlist[size++] = item;
            list = newlist;
        };
    
        T& operator[] (int index) {
            return list[index];
        };

        Array<T> slice (int s, int e) {
            Array<T> newarr;
            for (int i = s; i < e; i++) {
                newarr.push(list[i]);
            }
            return newarr;
        }

        T pop(int index) {
            T *newlist = new T[--size];
            T val;
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

        int remove (T value) {
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

        template <typename R> 
        Array<R> map (R (*f) (T) ) {
            Array<R> retarr;
            for (int i=0; i<size; i++) {
                retarr.push(f(list[i]));
            }
            return retarr;
        }

        Array<T> filter (bool (*f) (T, int) ) {
            Array<T> retarr;
            for (int i=0; i<size; i++) {
                if (f(list[i], i) == true) {
                    retarr.push(list[i]);
                }
            }
            return retarr;
        }

        T reduce (T (*f) (T, T, int)) {
            T agg = 0;
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
        void forEach(void (*f) (T)) {
            for (int i=0; i<size; i++) {
                f(list[i]);
            };
        };
        T max () {
            return this->reduce([](T li, T agg, int i) {return li > agg ? li : agg;});
        };
        T min () {
            return this->reduce([](T li, T agg, int i) {return li < agg ? li : agg;});
        };
};

template <typename T>
class Dictionary {
    Array<string> keys;
    Array<T> values;
    public:
        void add(string key, T value) {
            int index;
            for (int i=0; i<keys.size; i++) {
                if (key == keys[i]) {
                    index = keys.remove(key);
                    values.pop(index);
                }
            }
            keys.push(key);
            values.push(value);
        }

        T& operator[] (string key) {
            for (int i=0; i<keys.size; i++) {
                if (key == keys[i]) {
                    return values[i];
                }
            }
            return values[values.size+1];

        }

        void print( ) {
            cout << "{'" << keys[0] << "': " << values[0];
            for (int i=1; i<keys.size; i++) {
                cout << ", '" << keys[i] << "': " << values[i];
            }
            cout << "}\n";
        }
};
