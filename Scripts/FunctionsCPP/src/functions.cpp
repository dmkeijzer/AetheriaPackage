#include <iostream>
#include "makedic.cpp"

using namespace std;

int main ( ) {
    Array<int> arr;
    for (int i=0; i<30; i++) {
        arr.push(5*i);
    }
    arr.print();

    cout << "Removing 2nd term: " << arr.pop(1) << endl;

    cout << "Removing first twenty:\n";
    arr.remove(20);
    arr.print();

    cout << "\nNew Array:\n";
    Array<int> a;
    for (int j = 0; j < 20; j++) {
        a.push(j * j);
    }

    cout << "Joined: " << a.join(" | ") << endl;

    cout << "Subarray:\n";
    a.slice(0, 5).print();

    cout << "Sum: " << a.reduce([](int cur, int ag, int i) {return ag + cur;}) << " and double:" << endl;
    a.map<int>([](int cur) {return 2*cur;}).print();

    cout << "Even:" << endl;
    a.filter([](int cur, int ind) {return cur % 2 == 0;}).print();

    cout << "Odd:" << endl;
    a.filter([](int cur, int ind) {return cur % 2 != 0;}).print();

    cout << "\nDictionaries:\n";
    Dictionary<int> dict;
    string randomStrings[] = {"cool", "bad", "good", "ugly", "wonder"};

    for (unsigned long int j=1; j<1+sizeof(randomStrings) / sizeof(randomStrings[0]); j++) {
        dict.add(randomStrings[j-1], j);
    }
    dict.print();
    cout << dict["cool"] << endl;

    dict.add("cool", 100);
    dict["bad"] = 42;
    dict.print();
    return EXIT_SUCCESS;
};