#include <vector>
#include <utility>

class vectorHashIterator;

class vectorHash {
  std::size_t count;

public:
  std::vector<int> vec;
  vectorHash();
  int &operator[](std::size_t idx);
  vectorHashIterator begin();
  vectorHashIterator end();
  vectorHashIterator find(std::size_t idx);
  std::pair<vectorHashIterator, bool> insert(std::pair<unsigned int, int> p);
  void erase(vectorHashIterator iter);
  std::size_t size();
};

class vectorHashIterator {
private:
  vectorHash *vh;

public:
  unsigned int first;
  int second;
  void seek();
  vectorHashIterator(vectorHash *vh, int x);
  vectorHashIterator *operator->();
  vectorHashIterator operator++();
};

inline bool operator==(const vectorHashIterator &lhs,
                       const vectorHashIterator &rhs) {
  return lhs.first == rhs.first;
}
inline bool operator!=(const vectorHashIterator &lhs,
                       const vectorHashIterator &rhs) {
  return lhs.first != rhs.first;
}
