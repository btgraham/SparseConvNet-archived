#include "vectorHash.h"
//#include<iostream>
vectorHash::vectorHash() : count(0) {}
std::size_t vectorHash::size() { return count; }
int &vectorHash::operator[](std::size_t idx) {
  if (idx >= vec.size())
    vec.resize(idx + 1, -99);
  if (vec[idx] == -99)
    count++;
  return vec[idx];
}
vectorHashIterator vectorHash::begin() { return vectorHashIterator(this, 0); }
vectorHashIterator vectorHash::end() {
  return vectorHashIterator(this, vec.size());
}
vectorHashIterator vectorHash::find(std::size_t idx) {
  if (idx >= vec.size() or vec[idx] == -99) {
    return end();
  } else {
    return vectorHashIterator(this, idx);
  }
}
std::pair<vectorHashIterator, bool>
vectorHash::insert(std::pair<unsigned int, int> p) {
  if (p.first >= vec.size())
    vec.resize(p.first + 1, -99);
  if (vec[p.first] == -99) {
    count++;
    vec[p.first] = p.second;
    return std::make_pair(vectorHashIterator(this, p.first), true);
  } else {
    return std::make_pair(vectorHashIterator(this, p.first), false);
  }
}
void vectorHash::erase(vectorHashIterator iter) {
  vec[iter->first] = -99;
  count--;
}

void vectorHashIterator::seek() {
  while (first < vh->vec.size() and vh->vec[first] == -99)
    first++;
  if (first < vh->vec.size())
    second = vh->vec[first];
}
vectorHashIterator::vectorHashIterator(vectorHash *vh, int x) : vh(vh) {
  first = x;
  seek();
}
vectorHashIterator *vectorHashIterator::operator->() { return this; }
vectorHashIterator vectorHashIterator::operator++() {
  first++;
  seek();
  return *this;
}
