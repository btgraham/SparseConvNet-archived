#pragma once
#include <vector>
#include <glob.h>
#include <string>
//return vector 0,1,...,n-1
std::vector<int> range(int n);

//Integer powers
//http://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int  (sykora)
int ipow(int base, int exp);

//Calculate the number of points in a d-dimensional triangle/tetrahedro with given side length.
//Binomial coefficient \binom{linearSize+dimension-1}{dimension}
int triangleSize(int linearSize,int dimension);

//Assume test.size() is at least k.
template<typename t> std::vector<int> vectorTopIndices(std::vector<t> &test, int k);


std::vector<std::string> globVector(const std::string& pattern);
