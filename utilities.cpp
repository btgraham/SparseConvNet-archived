#include "utilities.h"

std::vector<int> range(int n) {
  std::vector<int> ret(n);
  for (int i=0; i<n; i++)
    ret[i]=i;
  return ret;
}

//http://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int  (sykora)
int ipow(int base, int exp)
{
  int result = 1;
  while (exp)
    {
      if (exp & 1)
        result *= base;
      exp >>= 1;
      base *= base;
    }

  return result;
}

int triangleSize(int linearSize,int dimension) {
  int fs=1;
  for (int i=1;i<=dimension;++i)
    fs=fs*(linearSize+dimension-i)/i;
  return fs;
}


//Assume test.size() is at least k, and k>=1.
template<typename t> std::vector<int> vectorTopIndices(std::vector<t> &test, int k) {
  std::vector<int> indices(k);
  std::vector<t> q(k);
  q[0]=test[0];
  for (int i=1;i<test.size();i++) {
    int j=std::min(i,k);
    if (test[i]>q[j-1]) {
      --j;
      for (;j>0 and test[i]>q[j-1];--j) {
        q[j]=q[j-1];
        indices[j]=indices[j-1];
      }
    }
    if (j<k) { q[j]=test[i]; indices[j]=i; }
  }
  return indices;
}
template std::vector<int> vectorTopIndices<float>(std::vector<float> &test, int k);
