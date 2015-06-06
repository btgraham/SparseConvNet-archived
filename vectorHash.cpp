#include "vectorHash.h"
//#include<iostream>
vectorHash::vectorHash() : count(0) {
}
std::size_t vectorHash::size() {
  return count;
}
int& vectorHash::operator[](std::size_t idx)  {
  if (idx>=vec.size())
    vec.resize(idx+1,-99);
  if (vec[idx]==-99) count++;
  return vec[idx];
}
vectorHashIterator vectorHash::begin() {
  return vectorHashIterator(*this,0);
}
vectorHashIterator vectorHash::end() {
  return vectorHashIterator(*this,vec.size());
}
vectorHashIterator vectorHash::find(std::size_t idx) {
  if (idx>=vec.size() or vec[idx]==-99) {
    return end();
  } else {
    return vectorHashIterator(*this,idx);
  }
}
void vectorHash::erase(vectorHashIterator iter) {
  vec[iter->first]=-99;
  count--;
}


void vectorHashIterator::seek() {
  while (first<vh.vec.size() and second==-99) {
    first++;
    second=vh[first];
  }
}
vectorHashIterator::vectorHashIterator(vectorHash &vh, int x) : vh(vh) {
  first=x;
  if (x<vh.size()) {
    second=vh.vec[x];
    seek();
  }
}
vectorHashIterator* vectorHashIterator::operator-> () {
  return this;
}
vectorHashIterator vectorHashIterator::operator++() {
  first++;
  second=vh.vec[first];
  seek();
  return *this;
}
