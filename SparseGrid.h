//We need a hash table:
//We can either use
// a) Google's sparsehash dense_hash_map, or
// b) the C++11 std::unordered_map
// Option (a) seems fastest. To speed up (b), use tbb::scalable_alloc (or similar) to prevent threads getting in each others way as they access memory to grow the maps.

#define USE_GOOGLE_SPARSEHASH

#pragma once
#include <stdint.h>

#ifdef USE_GOOGLE_SPARSEHASH
#include <functional>
#include <google/dense_hash_map>
typedef google::dense_hash_map<int64_t, int, std::hash<int64_t>, std::equal_to<int64_t> > SparseGridMap;

class SparseGrid {
public:
  int backgroundCol;
  SparseGridMap mp;
  SparseGrid() {
    backgroundCol=-1; //Indicate that no "null vector" is needed
    mp.set_empty_key(-99); // dense_hash_map needs an empty key that will not be used as a real key
    mp.set_deleted_key(-98);
    mp.min_load_factor(0.0f);

  }
};
#else
//Libraries -ltbb -ltbbmalloc
#include <unordered_map>
#include <tbb/scalable_allocator.h>
//typedef std::unordered_map<int64_t,int,std::hash<int64_t>,std::equal_to<int64_t>,std::allocator< std::pair<const int64_t,int> > > SparseGridMap;
typedef std::unordered_map<int64_t,int,std::hash<int64_t>,std::equal_to<int64_t>,tbb::scalable_allocator< std::pair<const int64_t,int> > > SparseGridMap;


class SparseGrid {
public:
  int backgroundCol; //Set to -1 when no "null vector" is needed
  SparseGridMap mp;
  SparseGrid() : backgroundCol(-1) {}
};
#endif
