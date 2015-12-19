#include "Regions.h"
#include "utilities.h"
#include <iostream>
#include <cassert>
#include <cmath>

RectangularRegions::RectangularRegions(int nIn, int nOut, int dimension, int s)
    : nIn(nIn), nOut(nOut), dimension(dimension), s(s) {
  sd = ipow(s, dimension);
}
RectangularRegions::~RectangularRegions() {}

RegularSquareRegions::RegularSquareRegions(int nIn, int nOut, int dimension,
                                           int poolSize, int poolStride)
    : RectangularRegions(nIn, nOut, dimension, poolSize), poolSize(poolSize),
      poolStride(poolStride) {
  assert(nIn == poolSize + (nOut - 1) * poolStride);
}
RegularSquareRegions::~RegularSquareRegions() {}
int RegularSquareRegions::inputL(int axis, int j) { return j * poolStride; }
int RegularSquareRegions::inputR(int axis, int j) {
  return j * poolStride + poolSize;
}
int RegularSquareRegions::outputL(int axis, int i) {
  return std::max(0, (i - poolSize + poolStride) / poolStride);
}
int RegularSquareRegions::outputR(int axis, int i) {
  return std::min(i / poolStride + 1, nOut);
}

PseudorandomOverlappingFmpTicks::PseudorandomOverlappingFmpTicks(int nIn,
                                                                 int nOut,
                                                                 int poolSize,
                                                                 RNG &rng) {
  assert(nIn >= nOut - 1 + poolSize);
  double alpha = (nIn - poolSize) * 1.0 / (nOut - 1);
  double u = rng.uniform(0, 1);
  for (int j = 0; j < nOut; ++j) {
    int i = (int)((j + u) * alpha) - (int)(u * alpha);
    inputL.push_back(i);
    inputR.push_back(i + poolSize);
  }
  assert(inputR.back() == nIn);
  outputL.resize(nIn, nOut);
  outputR.resize(nIn, 0);
  for (int i = 0; i < nOut; i++) {
    for (int j = inputL[i]; j < inputR[i]; j++) {
      outputL[j] = std::min(outputL[j], i);
      outputR[j] = std::max(outputR[j], i + 1);
    }
  }
}

PseudorandomNonOverlappingFmpTicks::PseudorandomNonOverlappingFmpTicks(
    int nIn, int nOut, int poolSize, RNG &rng) {
  double alpha = nIn * 1.0 / nOut;
  double u = rng.uniform(0, 1);
  assert(nIn >= nOut - 1 + poolSize);
  assert((int)ceil(alpha) == poolSize);
  for (int j = 0; j < nOut; ++j)
    inputL.push_back((int)((j + u) * alpha) - (int)(u * alpha));
  for (int j = 1; j <= nOut; ++j)
    inputR.push_back((int)((j + u) * alpha) - (int)(u * alpha));
  assert(inputR.back() == nIn);
  outputL.resize(nIn, nOut);
  outputR.resize(nIn, 0);
  for (int i = 0; i < nOut; i++) {
    for (int j = inputL[i]; j < inputR[i]; j++) {
      outputL[j] = std::min(outputL[j], i);
      outputR[j] = std::max(outputR[j], i + 1);
    }
  }
}

RandomOverlappingFmpTicks::RandomOverlappingFmpTicks(int nIn, int nOut,
                                                     int poolSize, RNG &rng) {
  assert(nIn > nOut);
  int alpha = nIn * 1.0 / nOut;
  std::vector<int> inc;
  inc.resize(nOut * (alpha + 1) - nIn, alpha);
  inc.resize(nOut - 1, alpha + 1);
  rng.vectorShuffle(inc);
  inputL.push_back(0);
  inputR.push_back(poolSize);
  for (int i = 0; i < nOut - 1; i++) {
    inputL.push_back(inputL.back() + inc[i]);
    inputR.push_back(inputL.back() + poolSize);
  }
  assert(inputR.back() == nIn);
  outputL.resize(nIn, nOut);
  outputR.resize(nIn, 0);
  for (int i = 0; i < nOut; i++) {
    for (int j = inputL[i]; j < inputR[i]; j++) {
      outputL[j] = std::min(outputL[j], i);
      outputR[j] = std::max(outputR[j], i + 1);
    }
  }
}

RandomNonOverlappingFmpTicks::RandomNonOverlappingFmpTicks(int nIn, int nOut,
                                                           int poolSize,
                                                           RNG &rng) {
  assert(nIn > nOut);
  int alpha = nIn * 1.0 / nOut;
  std::vector<int> inc;
  inc.resize(nOut * (alpha + 1) - nIn, alpha);
  inc.resize(nOut, alpha + 1);
  rng.vectorShuffle(inc);
  inputL.push_back(0);
  inputR.push_back(inc[0]);
  for (int i = 0; i < nOut - 1; i++) {
    inputL.push_back(inputL.back() + inc[i]);
    inputR.push_back(inputR.back() + inc[i + 1]);
  }
  assert(inputR.back() == nIn);
  for (int i = 0; i < nOut; i++) {
    for (int j = inputL[i]; j < inputR[i]; j++) {
      outputL.push_back(i);
      outputR.push_back(i + 1);
    }
  }
}

template <typename ticks>
FractionalPoolingRegions<ticks>::FractionalPoolingRegions(int nIn, int nOut,
                                                          int dimension,
                                                          int poolSize,
                                                          RNG &rng)
    : RectangularRegions(nIn, nOut, dimension, poolSize) {
  for (int i = 0; i < dimension; ++i)
    pb.push_back(ticks(nIn, nOut, poolSize, rng));
  assert(nIn >= nOut + poolSize - 1);
}
template <typename ticks>
FractionalPoolingRegions<ticks>::~FractionalPoolingRegions() {}

template <typename ticks>
int FractionalPoolingRegions<ticks>::inputL(int axis, int j) {
  return pb[axis].inputL[j];
}
template <typename ticks>
int FractionalPoolingRegions<ticks>::inputR(int axis, int j) {
  return pb[axis].inputR[j];
}
template <typename ticks>
int FractionalPoolingRegions<ticks>::outputL(int axis, int i) {
  return pb[axis].outputL[i];
}
template <typename ticks>
int FractionalPoolingRegions<ticks>::outputR(int axis, int i) {
  return pb[axis].outputR[i];
}

template class FractionalPoolingRegions<PseudorandomOverlappingFmpTicks>;
template class FractionalPoolingRegions<PseudorandomNonOverlappingFmpTicks>;
template class FractionalPoolingRegions<RandomOverlappingFmpTicks>;
template class FractionalPoolingRegions<RandomNonOverlappingFmpTicks>;

void gridRulesNonOverlapping(
    SparseGrid &inputGrid,  // Keys 0,1,...,powf(regions.nIn,dimension)-1
                            // represent grid points
    SparseGrid &outputGrid, // Keys 0,1,...,powf(regions.nOut,dimension)-1
                            // represent grid points
    RectangularRegions &regions, int &nOutputSpatialSites,
    std::vector<int> &rules) {
#ifdef USE_VECTOR_HASH
  outputGrid.mp.vec.resize(ipow(regions.nOut, regions.dimension), -99);
#endif
  switch (regions.dimension) {
  case 1:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = (iter->first);
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        int64_t outKey = j0;
        int rulesOffset = (i0 - regions.inputL(0, j0));
        auto f = outputGrid.mp.find(outKey);
        if (f == outputGrid.mp.end()) {
          f = outputGrid.mp.insert(std::make_pair(outKey,
                                                  nOutputSpatialSites++)).first;
          rules.resize(nOutputSpatialSites * regions.sd,
                       -1); // some -1s should be inputGrid.backgroundCol
          for (int ii0 = 0; ii0 < regions.inputR(0, j0) - regions.inputL(0, j0);
               ++ii0) {
            int k = ii0;
            rules[(nOutputSpatialSites - 1) * regions.sd + k] =
                inputGrid.backgroundCol;
          }
        }
        rules[f->second * regions.sd + rulesOffset] = iter->second;
      }
    }
    break;
  case 2:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn) % regions.nIn;
      int i1 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          int64_t outKey = (int64_t)j0 * regions.nOut + (int64_t)j1;
          int rulesOffset = (i0 - regions.inputL(0, j0)) * regions.s +
                            (i1 - regions.inputL(1, j1));
          auto f = outputGrid.mp.find(outKey);
          if (f == outputGrid.mp.end()) {
            f = outputGrid.mp.insert(std::make_pair(
                                         outKey, nOutputSpatialSites++)).first;
            rules.resize(nOutputSpatialSites * regions.sd,
                         -1); // some -1s should be inputGrid.backgroundCol

            for (int ii0 = 0;
                 ii0 < regions.inputR(0, j0) - regions.inputL(0, j0); ++ii0) {
              for (int ii1 = 0;
                   ii1 < regions.inputR(1, j1) - regions.inputL(1, j1); ++ii1) {
                int k = ii0 * regions.s + ii1;
                rules[(nOutputSpatialSites - 1) * regions.sd + k] =
                    inputGrid.backgroundCol;
              }
            }
          }
          rules[f->second * regions.sd + rulesOffset] = iter->second;
        }
      }
    }
    break;
  case 3:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i1 = ((iter->first) / regions.nIn) % regions.nIn;
      int i2 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          for (int j2 = regions.outputL(2, i2); j2 < regions.outputR(2, i2);
               ++j2) {
            int64_t outKey = (int64_t)j0 * regions.nOut * regions.nOut +
                             (int64_t)j1 * regions.nOut + (int64_t)j2;
            int rulesOffset =
                (i0 - regions.inputL(0, j0)) * regions.s * regions.s +
                (i1 - regions.inputL(1, j1)) * regions.s +
                (i2 - regions.inputL(2, j2));
            auto f = outputGrid.mp.find(outKey);
            if (f == outputGrid.mp.end()) {
              f = outputGrid.mp.insert(std::make_pair(outKey,
                                                      nOutputSpatialSites++))
                      .first;
              rules.resize(nOutputSpatialSites * regions.sd,
                           -1); // some -1s should be inputGrid.backgroundCol

              for (int ii0 = 0;
                   ii0 < regions.inputR(0, j0) - regions.inputL(0, j0); ++ii0) {
                for (int ii1 = 0;
                     ii1 < regions.inputR(1, j1) - regions.inputL(1, j1);
                     ++ii1) {
                  for (int ii2 = 0;
                       ii2 < regions.inputR(2, j2) - regions.inputL(2, j2);
                       ++ii2) {
                    int k = ii0 * regions.s * regions.s + ii1 * regions.s + ii2;
                    rules[(nOutputSpatialSites - 1) * regions.sd + k] =
                        inputGrid.backgroundCol;
                  }
                }
              }
            }
            rules[f->second * regions.sd + rulesOffset] = iter->second;
          }
        }
      }
    }
    break;
  case 4:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first / regions.nIn / regions.nIn / regions.nIn;
      int i1 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i2 = ((iter->first) / regions.nIn) % regions.nIn;
      int i3 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          for (int j2 = regions.outputL(2, i2); j2 < regions.outputR(2, i2);
               ++j2) {
            for (int j3 = regions.outputL(3, i3); j3 < regions.outputR(3, i3);
                 ++j3) {
              int64_t outKey =
                  (int64_t)j0 * regions.nOut * regions.nOut * regions.nOut +
                  (int64_t)j1 * regions.nOut * regions.nOut +
                  (int64_t)j2 * regions.nOut + (int64_t)j3;
              int rulesOffset =
                  (i0 - regions.inputL(0, j0)) * regions.s * regions.s *
                      regions.s +
                  (i1 - regions.inputL(1, j1)) * regions.s * regions.s +
                  (i2 - regions.inputL(2, j2)) * regions.s +
                  (i3 - regions.inputL(3, j3));
              auto f = outputGrid.mp.find(outKey);
              if (f == outputGrid.mp.end()) {
                f = outputGrid.mp.insert(std::make_pair(outKey,
                                                        nOutputSpatialSites++))
                        .first;
                rules.resize(nOutputSpatialSites * regions.sd,
                             -1); // some -1s should be inputGrid.backgroundCol
                for (int ii0 = 0;
                     ii0 < regions.inputR(0, j0) - regions.inputL(0, j0);
                     ++ii0) {
                  for (int ii1 = 0;
                       ii1 < regions.inputR(1, j1) - regions.inputL(1, j1);
                       ++ii1) {
                    for (int ii2 = 0;
                         ii2 < regions.inputR(2, j2) - regions.inputL(2, j2);
                         ++ii2) {
                      for (int ii3 = 0;
                           ii3 < regions.inputR(3, j3) - regions.inputL(3, j3);
                           ++ii3) {
                        int k = ii0 * regions.s * regions.s * regions.s +
                                ii1 * regions.s * regions.s + ii2 * regions.s +
                                ii3;
                        rules[(nOutputSpatialSites - 1) * regions.sd + k] =
                            inputGrid.backgroundCol;
                      }
                    }
                  }
                }
              }
              rules[f->second * regions.sd + rulesOffset] = iter->second;
            }
          }
        }
      }
    }
    break;
  }
  if (outputGrid.mp.size() <
      ipow(regions.nOut, regions.dimension)) { // Null vector/background needed
    for (int i = 0; i < regions.sd; ++i)
      rules.push_back(inputGrid.backgroundCol);
    outputGrid.backgroundCol = nOutputSpatialSites++;
  }
}

void gridRulesOverlappingMin(
    SparseGrid &inputGrid,  // Keys 0,1,...,powf(regions.nIn,dimension)-1
                            // represent grid points
    SparseGrid &outputGrid, // Keys 0,1,...,powf(regions.nOut,dimension)-1
                            // represent grid points
    RectangularRegions &regions, int &nOutputSpatialSites,
    std::vector<int> &rules, int minActiveInputs) {
#ifdef USE_VECTOR_HASH
  outputGrid.mp.vec.resize(ipow(regions.nOut, regions.dimension), -99);
#endif
  switch (regions.dimension) {
  case 1:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = (iter->first);
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        int64_t outKey = (int64_t)j0;
        if (outputGrid.mp.find(outKey) ==
            outputGrid.mp.end()) { // Add line to rules
          int activeInputCtr = 0;
          for (int ii0 = regions.inputL(0, j0); ii0 < regions.inputR(0, j0);
               ++ii0) {
            int64_t inKey = (int64_t)ii0;
            auto iter2 = inputGrid.mp.find(inKey);
            if (iter2 == inputGrid.mp.end()) {
              rules.push_back(inputGrid.backgroundCol);
            } else {
              rules.push_back(iter2->second);
              activeInputCtr++;
            }
          }
          if (activeInputCtr >= minActiveInputs) {
            outputGrid.mp[outKey] = nOutputSpatialSites++;
          } else {
            outputGrid.mp[outKey] = -2;
            rules.resize(nOutputSpatialSites * regions.sd);
          }
        }
      }
    }
    break;
  case 2:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn) % regions.nIn;
      int i1 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          int64_t outKey = (int64_t)j0 * regions.nOut + (int64_t)j1;
          if (outputGrid.mp.find(outKey) ==
              outputGrid.mp.end()) { // Add line to rules
            int activeInputCtr = 0;
            for (int ii0 = regions.inputL(0, j0); ii0 < regions.inputR(0, j0);
                 ++ii0) {
              for (int ii1 = regions.inputL(1, j1); ii1 < regions.inputR(1, j1);
                   ++ii1) {
                int64_t inKey = (int64_t)ii0 * regions.nIn + (int64_t)ii1;
                auto iter2 = inputGrid.mp.find(inKey);
                if (iter2 == inputGrid.mp.end()) {
                  rules.push_back(inputGrid.backgroundCol);
                } else {
                  rules.push_back(iter2->second);
                  activeInputCtr++;
                }
              }
            }
            if (activeInputCtr >= minActiveInputs) {
              outputGrid.mp[outKey] = nOutputSpatialSites++;
            } else {
              outputGrid.mp[outKey] = -2;
              rules.resize(nOutputSpatialSites * regions.sd);
            }
          }
        }
      }
    }
    break;
  case 3:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i1 = ((iter->first) / regions.nIn) % regions.nIn;
      int i2 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          for (int j2 = regions.outputL(2, i2); j2 < regions.outputR(2, i2);
               ++j2) {
            int64_t outKey = (int64_t)j0 * regions.nOut * regions.nOut +
                             (int64_t)j1 * regions.nOut + (int64_t)j2;
            if (outputGrid.mp.find(outKey) ==
                outputGrid.mp.end()) { // Add line to rules
              int activeInputCtr = 0;
              for (int ii0 = regions.inputL(0, j0); ii0 < regions.inputR(0, j0);
                   ++ii0) {
                for (int ii1 = regions.inputL(1, j1);
                     ii1 < regions.inputR(1, j1); ++ii1) {
                  for (int ii2 = regions.inputL(2, j2);
                       ii2 < regions.inputR(2, j2); ++ii2) {
                    int64_t inKey = (int64_t)ii0 * regions.nIn * regions.nIn +
                                    (int64_t)ii1 * regions.nIn + (int64_t)ii2;
                    auto iter2 = inputGrid.mp.find(inKey);
                    if (iter2 == inputGrid.mp.end()) {
                      rules.push_back(inputGrid.backgroundCol);
                    } else {
                      rules.push_back(iter2->second);
                      activeInputCtr++;
                    }
                  }
                }
              }
              if (activeInputCtr >= minActiveInputs) {
                outputGrid.mp[outKey] = nOutputSpatialSites++;
              } else {
                outputGrid.mp[outKey] = -2;
                rules.resize(nOutputSpatialSites * regions.sd);
              }
            }
          }
        }
      }
    }
    break;
  case 4:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first / regions.nIn / regions.nIn / regions.nIn;
      int i1 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i2 = ((iter->first) / regions.nIn) % regions.nIn;
      int i3 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          for (int j2 = regions.outputL(2, i2); j2 < regions.outputR(2, i2);
               ++j2) {
            for (int j3 = regions.outputL(3, i3); j3 < regions.outputR(3, i3);
                 ++j3) {
              int64_t outKey =
                  (int64_t)j0 * regions.nOut * regions.nOut * regions.nOut +
                  (int64_t)j1 * regions.nOut * regions.nOut +
                  (int64_t)j2 * regions.nOut + (int64_t)j3;
              if (outputGrid.mp.find(outKey) ==
                  outputGrid.mp.end()) { // Add line to rules
                int activeInputCtr = 0;
                for (int ii0 = regions.inputL(0, j0);
                     ii0 < regions.inputR(0, j0); ++ii0) {
                  for (int ii1 = regions.inputL(1, j1);
                       ii1 < regions.inputR(1, j1); ++ii1) {
                    for (int ii2 = regions.inputL(2, j2);
                         ii2 < regions.inputR(2, j2); ++ii2) {
                      for (int ii3 = regions.inputL(3, j3);
                           ii3 < regions.inputR(3, j3); ++ii3) {
                        int64_t inKey =
                            (int64_t)ii0 * regions.nIn * regions.nIn *
                                regions.nIn +
                            (int64_t)ii1 * regions.nIn * regions.nIn +
                            (int64_t)ii2 * regions.nIn + (int64_t)ii3;
                        auto iter2 = inputGrid.mp.find(inKey);
                        if (iter2 == inputGrid.mp.end()) {
                          rules.push_back(inputGrid.backgroundCol);
                        } else {
                          rules.push_back(iter2->second);
                          activeInputCtr++;
                        }
                      }
                    }
                  }
                }
                if (activeInputCtr >= minActiveInputs) {
                  outputGrid.mp[outKey] = nOutputSpatialSites++;
                } else {
                  outputGrid.mp[outKey] = -2;
                  rules.resize(nOutputSpatialSites * regions.sd);
                }
              }
            }
          }
        }
      }
    }
    break;
  }
  for (auto iter = outputGrid.mp.begin(); iter != outputGrid.mp.end(); ++iter)
    if (iter->second == -2)
      outputGrid.mp.erase(iter);
  if (outputGrid.mp.size() <
      ipow(regions.nOut, regions.dimension)) { // Null vector/background needed
    for (int i = 0; i < regions.sd; ++i)
      rules.push_back(inputGrid.backgroundCol);
    outputGrid.backgroundCol = nOutputSpatialSites++;
  }
}

void gridRulesOverlappingNoMin(
    SparseGrid &inputGrid,  // Keys 0,1,...,powf(regions.nIn,dimension)-1
                            // represent grid points
    SparseGrid &outputGrid, // Keys 0,1,...,powf(regions.nOut,dimension)-1
                            // represent grid points
    RectangularRegions &regions, int &nOutputSpatialSites,
    std::vector<int> &rules) {
#ifdef USE_VECTOR_HASH
  outputGrid.mp.vec.resize(ipow(regions.nOut, regions.dimension), -99);
#endif
  switch (regions.dimension) {
  case 1:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = (iter->first);
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        int64_t outKey = j0;
        int rulesOffset = (i0 - regions.inputL(0, j0));
        auto f = outputGrid.mp.find(outKey);
        if (f == outputGrid.mp.end()) {
          f = outputGrid.mp.insert(std::make_pair(outKey,
                                                  nOutputSpatialSites++)).first;
          rules.resize(nOutputSpatialSites * regions.sd,
                       inputGrid.backgroundCol);
        }
        rules[f->second * regions.sd + rulesOffset] = iter->second;
      }
    }
    break;
  case 2:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn) % regions.nIn;
      int i1 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          int64_t outKey = (int64_t)j0 * regions.nOut + (int64_t)j1;
          int rulesOffset = (i0 - regions.inputL(0, j0)) * regions.s +
                            (i1 - regions.inputL(1, j1));
          auto f = outputGrid.mp.find(outKey);
          if (f == outputGrid.mp.end()) {
            f = outputGrid.mp.insert(std::make_pair(
                                         outKey, nOutputSpatialSites++)).first;
            rules.resize(nOutputSpatialSites * regions.sd,
                         inputGrid.backgroundCol);
          }
          rules[f->second * regions.sd + rulesOffset] = iter->second;
        }
      }
    }
    break;
  case 3:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i1 = ((iter->first) / regions.nIn) % regions.nIn;
      int i2 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i0); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          for (int j2 = regions.outputL(2, i2); j2 < regions.outputR(2, i2);
               ++j2) {
            int64_t outKey = (int64_t)j0 * regions.nOut * regions.nOut +
                             (int64_t)j1 * regions.nOut + (int64_t)j2;
            int rulesOffset =
                (i0 - regions.inputL(0, j0)) * regions.s * regions.s +
                (i1 - regions.inputL(1, j1)) * regions.s +
                (i2 - regions.inputL(2, j2));
            auto f = outputGrid.mp.find(outKey);
            if (f == outputGrid.mp.end()) {
              f = outputGrid.mp.insert(std::make_pair(outKey,
                                                      nOutputSpatialSites++))
                      .first;
              rules.resize(nOutputSpatialSites * regions.sd,
                           inputGrid.backgroundCol);
            }
            rules[f->second * regions.sd + rulesOffset] = iter->second;
          }
        }
      }
    }
    break;
  case 4:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first / regions.nIn / regions.nIn / regions.nIn;
      int i1 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i2 = ((iter->first) / regions.nIn) % regions.nIn;
      int i3 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(0, i1); j0 < regions.outputR(0, i0); ++j0) {
        for (int j1 = regions.outputL(1, i1); j1 < regions.outputR(1, i1);
             ++j1) {
          for (int j2 = regions.outputL(2, i2); j2 < regions.outputR(2, i2);
               ++j2) {
            for (int j3 = regions.outputL(3, i3); j3 < regions.outputR(3, i3);
                 ++j3) {
              int64_t outKey =
                  (int64_t)j0 * regions.nOut * regions.nOut * regions.nOut +
                  (int64_t)j1 * regions.nOut * regions.nOut +
                  (int64_t)j2 * regions.nOut + (int64_t)j3;
              int rulesOffset =
                  (i0 - regions.inputL(0, j0)) * regions.s * regions.s *
                      regions.s +
                  (i1 - regions.inputL(1, j1)) * regions.s * regions.s +
                  (i2 - regions.inputL(2, j2)) * regions.s +
                  (i3 - regions.inputL(3, j3));
              auto f = outputGrid.mp.find(outKey);
              if (f == outputGrid.mp.end()) {
                f = outputGrid.mp.insert(std::make_pair(outKey,
                                                        nOutputSpatialSites++))
                        .first;
                rules.resize(nOutputSpatialSites * regions.sd,
                             inputGrid.backgroundCol);
              }
              rules[f->second * regions.sd + rulesOffset] = iter->second;
            }
          }
        }
      }
    }
    break;
  }
  if (outputGrid.mp.size() <
      ipow(regions.nOut, regions.dimension)) { // Null vector/background needed
    for (int i = 0; i < regions.sd; ++i)
      rules.push_back(inputGrid.backgroundCol);
    outputGrid.backgroundCol = nOutputSpatialSites++;
  }
}

void gridRules(SparseGrid &inputGrid,  // Keys
                                       // 0,1,...,powf(regions.nIn,dimension)-1
                                       // represent grid points
               SparseGrid &outputGrid, // Keys
                                       // 0,1,...,powf(regions.nOut,dimension)-1
                                       // represent grid points
               RectangularRegions &regions, int &nOutputSpatialSites,
               std::vector<int> &rules, bool uniformSizeRegions,
               int minActiveInputs) {
  if (uniformSizeRegions and minActiveInputs == 1)
    gridRulesOverlappingNoMin(inputGrid, outputGrid, regions,
                              nOutputSpatialSites, rules);
  if (uniformSizeRegions and minActiveInputs > 1)
    gridRulesOverlappingMin(inputGrid, outputGrid, regions, nOutputSpatialSites,
                            rules, minActiveInputs);
  if (not uniformSizeRegions)
    gridRulesNonOverlapping(inputGrid, outputGrid, regions, nOutputSpatialSites,
                            rules);
}

RegularTriangularRegions::RegularTriangularRegions(int nIn, int nOut,
                                                   int dimension, int poolSize,
                                                   int poolStride)
    : nIn(nIn), nOut(nOut), dimension(dimension), s(poolSize),
      poolSize(poolSize), poolStride(poolStride) {
  S = 0; // Calculate #points in the triangular filter, and order them
  ord.resize(ipow(s, dimension), -1); // iterate over the s^d cube, -1 will mean
                                      // "not in the triangular region"
  for (int i = 0; i < ord.size(); i++) {
    int j = i, J = 0;
    while (j > 0) {
      J += j % s; // Calulate the L1-distance (Taxicab/Manhatten distance) from
                  // the "top left" corner
      j /= s;
    }
    if (J <
        s) // if the points lies in the triangle/pyramid, add it to the filter
      ord[i] = S++;
  }
  assert(nIn == poolSize + (nOut - 1) * poolStride);
}

int RegularTriangularRegions::inputL(int j) { return j * poolStride; }
int RegularTriangularRegions::outputL(int i) {
  return std::max(0, (i - poolSize + poolStride) / poolStride);
}
int RegularTriangularRegions::outputR(int i) {
  return std::min(i / poolStride + 1, nOut);
}

void gridRulesNoMin(
    SparseGrid &inputGrid,  // Keys 0,1,...,powf(regions.nIn,3)-1 represent grid
                            // points (plus paddding to form a square/cube); key
                            // -1 represents null/background vector
    SparseGrid &outputGrid, // Keys 0,1,...,powf(regions.nOut,3)-1 represent
                            // grid points (plus paddding to form a
                            // square/cube); key -1 represents null/background
                            // vector
    RegularTriangularRegions &regions, int &nOutputSpatialSites,
    std::vector<int> &rules) {
#ifdef USE_VECTOR_HASH
  outputGrid.mp.vec.resize(triangleSize(regions.nOut, regions.dimension), -99);
#endif
  switch (regions.dimension) {
  case 1:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = (iter->first);
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        int64_t outKey = j0;
        auto f = outputGrid.mp.find(outKey);
        if (f == outputGrid.mp.end()) {
          f = outputGrid.mp.insert(std::make_pair(outKey,
                                                  nOutputSpatialSites++)).first;
          rules.resize(nOutputSpatialSites * regions.S,
                       inputGrid.backgroundCol);
        }
        int rulesOffset = regions.ord[(i0 - regions.inputL(j0))];
        if (rulesOffset > -1)
          rules[f->second * regions.S + rulesOffset] = iter->second;
      }
    }
  case 2:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn) % regions.nIn;
      int i1 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        for (int j1 = regions.outputL(i1); j1 < regions.outputR(i1); ++j1) {
          if (j0 + j1 < regions.nOut) {
            int64_t outKey = (int64_t)j0 * regions.nOut + (int64_t)j1;
            auto f = outputGrid.mp.find(outKey);
            if (f == outputGrid.mp.end()) {
              f = outputGrid.mp.insert(std::make_pair(outKey,
                                                      nOutputSpatialSites++))
                      .first;
              rules.resize(nOutputSpatialSites * regions.S,
                           inputGrid.backgroundCol);
            }
            int rulesOffset =
                regions.ord[(i0 - regions.inputL(j0)) * regions.s +
                            (i1 - regions.inputL(j1))];
            if (rulesOffset > -1)
              rules[f->second * regions.S + rulesOffset] = iter->second;
          }
        }
      }
    }
    break;
  case 3:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i1 = ((iter->first) / regions.nIn) % regions.nIn;
      int i2 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        for (int j1 = regions.outputL(i1); j1 < regions.outputR(i1); ++j1) {
          for (int j2 = regions.outputL(i2); j2 < regions.outputR(i2); ++j2) {
            if (j0 + j1 + j2 < regions.nOut) {
              int64_t outKey = (int64_t)j0 * regions.nOut * regions.nOut +
                               (int64_t)j1 * regions.nOut + (int64_t)j2;
              auto f = outputGrid.mp.find(outKey);
              if (f == outputGrid.mp.end()) {
                f = outputGrid.mp.insert(std::make_pair(outKey,
                                                        nOutputSpatialSites++))
                        .first;
                rules.resize(nOutputSpatialSites * regions.S,
                             inputGrid.backgroundCol);
              }
              int rulesOffset =
                  regions
                      .ord[(i0 - regions.inputL(j0)) * regions.s * regions.s +
                           (i1 - regions.inputL(j1)) * regions.s +
                           (i2 - regions.inputL(j2))];
              if (rulesOffset > -1)
                rules[f->second * regions.S + rulesOffset] = iter->second;
            }
          }
        }
      }
    }
    break;
  case 4:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first / regions.nIn / regions.nIn / regions.nIn;
      int i1 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i2 = ((iter->first) / regions.nIn) % regions.nIn;
      int i3 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        for (int j1 = regions.outputL(i1); j1 < regions.outputR(i1); ++j1) {
          for (int j2 = regions.outputL(i2); j2 < regions.outputR(i2); ++j2) {
            for (int j3 = regions.outputL(i3); j3 < regions.outputR(i3); ++j3) {
              if (j0 + j1 + j2 + j3 < regions.nOut) {
                int64_t outKey =
                    (int64_t)j0 * regions.nOut * regions.nOut * regions.nOut +
                    (int64_t)j1 * regions.nOut * regions.nOut +
                    (int64_t)j2 * regions.nOut + (int64_t)j3;
                auto f = outputGrid.mp.find(outKey);
                if (f == outputGrid.mp.end()) {
                  f = outputGrid.mp.insert(std::make_pair(
                                               outKey, nOutputSpatialSites++))
                          .first;
                  rules.resize(nOutputSpatialSites * regions.S,
                               inputGrid.backgroundCol);
                }
                int rulesOffset =
                    regions
                        .ord[(i0 - regions.inputL(j0)) * regions.s * regions.s *
                                 regions.s +
                             (i1 - regions.inputL(j1)) * regions.s * regions.s +
                             (i2 - regions.inputL(j2)) * regions.s +
                             (i3 - regions.inputL(j3))];
                if (rulesOffset > -1)
                  rules[f->second * regions.S + rulesOffset] = iter->second;
              }
            }
          }
        }
      }
    }
    break;
  }
  if (outputGrid.mp.size() <
      triangleSize(regions.nOut,
                   regions.dimension)) { // Null vector/background needed
    for (int i = 0; i < regions.S; ++i)
      rules.push_back(inputGrid.backgroundCol);
    outputGrid.backgroundCol = nOutputSpatialSites++;
  }
}

void gridRulesMin(SparseGrid &inputGrid, // Keys 0,1,...,powf(regions.nIn,3)-1
                                         // represent grid points (plus paddding
                                         // to form a square/cube); key -1
                                         // represents null/background vector
                  SparseGrid &outputGrid, // Keys 0,1,...,powf(regions.nOut,3)-1
                                          // represent grid points (plus
                                          // paddding to form a square/cube);
                                          // key -1 represents null/background
                                          // vector
                  RegularTriangularRegions &regions, int &nOutputSpatialSites,
                  std::vector<int> &rules, int minActiveInputs) {
#ifdef USE_VECTOR_HASH
  outputGrid.mp.vec.resize(triangleSize(regions.nOut, regions.dimension), -99);
#endif
  switch (regions.dimension) {
  case 1:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first;
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        int64_t outKey = (int64_t)j0;
        if (outputGrid.mp.find(outKey) ==
            outputGrid.mp.end()) { // Add line to rules
          int activeInputCtr = 0;
          for (int ii0 = 0; ii0 < regions.s; ++ii0) {
            int64_t inKey = (int64_t)(regions.inputL(j0) + ii0);
            auto iter2 = inputGrid.mp.find(inKey);
            if (iter2 == inputGrid.mp.end()) {
              rules.push_back(inputGrid.backgroundCol);
            } else {
              rules.push_back(iter2->second);
              activeInputCtr++;
            }
          }
          if (activeInputCtr >= minActiveInputs) {
            outputGrid.mp[outKey] = nOutputSpatialSites++;
          } else {
            outputGrid.mp[outKey] = -2;
            rules.resize(nOutputSpatialSites * regions.S);
          }
        }
      }
    }
    break;
  case 2:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first / regions.nIn;
      int i1 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        for (int j1 = regions.outputL(i1); j1 < regions.outputR(i1); ++j1) {
          if (j0 + j1 < regions.nOut) {
            int64_t outKey = (int64_t)j0 * regions.nOut + (int64_t)j1;
            if (outputGrid.mp.find(outKey) ==
                outputGrid.mp.end()) { // Add line to rules
              int activeInputCtr = 0;
              for (int ii0 = 0; ii0 < regions.s; ++ii0) {
                for (int ii1 = 0; ii1 < regions.s - ii0; ++ii1) {
                  int64_t inKey =
                      (int64_t)(regions.inputL(j0) + ii0) * regions.nIn +
                      (int64_t)(regions.inputL(j1) + ii1);
                  auto iter2 = inputGrid.mp.find(inKey);
                  if (iter2 == inputGrid.mp.end()) {
                    rules.push_back(inputGrid.backgroundCol);
                  } else {
                    rules.push_back(iter2->second);
                    activeInputCtr++;
                  }
                }
              }
              if (activeInputCtr >= minActiveInputs) {
                outputGrid.mp[outKey] = nOutputSpatialSites++;
              } else {
                outputGrid.mp[outKey] = -2;
                rules.resize(nOutputSpatialSites * regions.S);
              }
            }
          }
        }
      }
    }
    break;
  case 3:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first / regions.nIn / regions.nIn;
      int i1 = ((iter->first) / regions.nIn) % regions.nIn;
      int i2 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        for (int j1 = regions.outputL(i1); j1 < regions.outputR(i1); ++j1) {
          for (int j2 = regions.outputL(i2); j2 < regions.outputR(i2); ++j2) {
            if (j0 + j1 + j2 < regions.nOut) {
              int64_t outKey = (int64_t)j0 * regions.nOut * regions.nOut +
                               (int64_t)j1 * regions.nOut + (int64_t)j2;
              if (outputGrid.mp.find(outKey) ==
                  outputGrid.mp.end()) { // Add line to rules
                int activeInputCtr = 0;
                for (int ii0 = 0; ii0 < regions.s; ++ii0) {
                  for (int ii1 = 0; ii1 < regions.s - ii0; ++ii1) {
                    for (int ii2 = 0; ii2 < regions.s - ii0 - ii1; ++ii2) {
                      int64_t inKey =
                          (int64_t)(regions.inputL(j0) + ii0) * regions.nIn *
                              regions.nIn +
                          (int64_t)(regions.inputL(j1) + ii1) * regions.nIn +
                          (int64_t)(regions.inputL(j2) + ii2);
                      auto iter2 = inputGrid.mp.find(inKey);
                      if (iter2 == inputGrid.mp.end()) {
                        rules.push_back(inputGrid.backgroundCol);
                      } else {
                        rules.push_back(iter2->second);
                        activeInputCtr++;
                      }
                    }
                  }
                }
                if (activeInputCtr >= minActiveInputs) {
                  outputGrid.mp[outKey] = nOutputSpatialSites++;
                } else {
                  outputGrid.mp[outKey] = -2;
                  rules.resize(nOutputSpatialSites * regions.S);
                }
              }
            }
          }
        }
      }
    }
    break;
  case 4:
    for (auto iter = inputGrid.mp.begin(); iter != inputGrid.mp.end(); ++iter) {
      int i0 = iter->first / regions.nIn / regions.nIn / regions.nIn;
      int i1 = ((iter->first) / regions.nIn / regions.nIn) % regions.nIn;
      int i2 = ((iter->first) / regions.nIn) % regions.nIn;
      int i3 = (iter->first) % regions.nIn;
      for (int j0 = regions.outputL(i0); j0 < regions.outputR(i0); ++j0) {
        for (int j1 = regions.outputL(i1); j1 < regions.outputR(i1); ++j1) {
          for (int j2 = regions.outputL(i2); j2 < regions.outputR(i2); ++j2) {
            for (int j3 = regions.outputL(i3); j3 < regions.outputR(i3); ++j3) {
              if (j0 + j1 + j2 + j3 < regions.nOut) {
                int64_t outKey =
                    (int64_t)j0 * regions.nOut * regions.nOut * regions.nOut +
                    (int64_t)j1 * regions.nOut * regions.nOut +
                    (int64_t)j2 * regions.nOut + (int64_t)j3;
                if (outputGrid.mp.find(outKey) ==
                    outputGrid.mp.end()) { // Add line to rules
                  int activeInputCtr = 0;
                  for (int ii0 = 0; ii0 < regions.s; ++ii0) {
                    for (int ii1 = 0; ii1 < regions.s - ii0; ++ii1) {
                      for (int ii2 = 0; ii2 < regions.s - ii0 - ii1; ++ii2) {
                        for (int ii3 = 0; ii3 < regions.s - ii0 - ii1 - ii2;
                             ++ii3) {
                          int64_t inKey = (int64_t)(regions.inputL(j0) + ii0) *
                                              regions.nIn * regions.nIn *
                                              regions.nIn +
                                          (int64_t)(regions.inputL(j1) + ii1) *
                                              regions.nIn * regions.nIn +
                                          (int64_t)(regions.inputL(j2) + ii2) *
                                              regions.nIn +
                                          (int64_t)(regions.inputL(j3) + ii3);
                          auto iter2 = inputGrid.mp.find(inKey);
                          if (iter2 == inputGrid.mp.end()) {
                            rules.push_back(inputGrid.backgroundCol);
                          } else {
                            rules.push_back(iter2->second);
                            activeInputCtr++;
                          }
                        }
                      }
                    }
                  }
                  if (activeInputCtr >= minActiveInputs) {
                    outputGrid.mp[outKey] = nOutputSpatialSites++;
                  } else {
                    outputGrid.mp[outKey] = -2;
                    rules.resize(nOutputSpatialSites * regions.S);
                  }
                }
              }
            }
          }
        }
      }
    }
    break;
  }
  for (auto iter = outputGrid.mp.begin(); iter != outputGrid.mp.end(); ++iter)
    if (iter->second == -2)
      outputGrid.mp.erase(iter);
  if (outputGrid.mp.size() <
      triangleSize(regions.nOut,
                   regions.dimension)) { // Null vector/background needed
    for (int i = 0; i < regions.S; ++i)
      rules.push_back(inputGrid.backgroundCol);
    outputGrid.backgroundCol = nOutputSpatialSites++;
  }
}

void gridRules(SparseGrid &inputGrid, SparseGrid &outputGrid,
               RegularTriangularRegions &regions, int &nOutputSpatialSites,
               std::vector<int> &rules, int minActiveInputs) {
  if (minActiveInputs == 1)
    gridRulesNoMin(inputGrid, outputGrid, regions, nOutputSpatialSites, rules);
  else
    gridRulesMin(inputGrid, outputGrid, regions, nOutputSpatialSites, rules,
                 minActiveInputs);
}
