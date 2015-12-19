// We need a sparse n-dimensional table
// We can either use

// a) Google's sparsehash dense_hash_map, or
#define USE_GOOGLE_SPARSEHASH

// b) the C++11 std::unordered_map
// #define USE_UNORDERED_MAP

// c) the C++11 std::unordered_map with tbb::scalable_alloc (or similar) to
// prevent threads getting in each others way as they access memory to grow the
// maps,
// #define USE_UNORDERED_MAP

// d) vectors disguised as a hash map (ok in 2 dimensions)
// #define USE_VECTOR_HASH

#pragma once
#include <stdint.h>

#ifdef USE_GOOGLE_SPARSEHASH
#include <functional>
#include <google/dense_hash_map>
typedef google::dense_hash_map<int64_t, int, std::hash<int64_t>,
                               std::equal_to<int64_t>> SparseGridMap;
class SparseGrid {
public:
  int backgroundCol;
  SparseGridMap mp;
  SparseGrid() {
    backgroundCol = -1;    // Indicate that no "null vector" is needed
    mp.set_empty_key(-99); // dense_hash_map needs an empty key that will not be
                           // used as a real key
    mp.set_deleted_key(-98); // and another for deleting
    mp.min_load_factor(0.0f);
  }
};
#endif

#ifdef USE_UNORDERED_MAP
#include <unordered_map>
typedef std::unordered_map<
    int64_t, int, std::hash<int64_t>, std::equal_to<int64_t>,
    std::allocator<std::pair<const int64_t, int>>> SparseGridMap;
class SparseGrid {
public:
  int backgroundCol; // Set to -1 when no "null vector" is needed
  SparseGridMap mp;
  SparseGrid() : backgroundCol(-1) {}
};
#endif

#ifdef USE_UNORDERED_MAP_TBB
// Libraries -ltbb -ltbbmalloc
#include <unordered_map>
#include <tbb/scalable_allocator.h>
typedef std::unordered_map<
    int64_t, int, std::hash<int64_t>, std::equal_to<int64_t>,
    tbb::scalable_allocator<std::pair<const int64_t, int>>> SparseGridMap;
class SparseGrid {
public:
  int backgroundCol; // Set to -1 when no "null vector" is needed
  SparseGridMap mp;
  SparseGrid() : backgroundCol(-1) {}
};
#endif

#ifdef USE_VECTOR_HASH
#include "vectorHash.h"
typedef vectorHash SparseGridMap;
class SparseGrid {
public:
  int backgroundCol; // Set to -1 when no "null vector" is needed
  SparseGridMap mp;
  SparseGrid() : backgroundCol(-1) {}
};
#endif
