// Supports 1,2,3 and 4 dimensions. Could extend to 5d, etc
// Supports convolutions and max-pooling with fixed size and stride
// Supports fractional max-pooling
// Supports triangular/tetrahedral lattices
#pragma once
#include <vector>
#include "Rng.h"
#include "SparseGrid.h"

class RectangularRegions {
public:
  int nIn;
  int nOut;
  int dimension;
  int s;
  int sd;
  RectangularRegions(int nIn, int nOut, int dimension, int s);
  ~RectangularRegions();
  // up to 4d - could extend to 5d ...
  // given a point in the output layer, location of "top-left" and
  // "bottom-right"
  // corner of the corresponding pooling region in the input layer
  // (left<=x<right)
  virtual int inputL(int axis, int j) = 0;
  virtual int inputR(int axis, int j) = 0;
  // given an input-layer point, what is the range of points in the output
  // layer to which it can be pooled (inclusive bounds)
  virtual int outputL(int axis, int i) = 0;
  virtual int outputR(int axis, int i) = 0;
};

class RegularSquareRegions : public RectangularRegions {
public:
  int poolSize;
  int poolStride;
  RegularSquareRegions(int nIn, int nOut, int dimension, int poolSize,
                       int poolStride);
  ~RegularSquareRegions();
  int inputL(int axis, int j);
  int inputR(int axis, int j);
  int outputL(int axis, int i);
  int outputR(int axis, int i);
};

class FractionalMaxPoolingTicks {
public:
  std::vector<int> inputL;
  std::vector<int> inputR;
  std::vector<int> outputL;
  std::vector<int> outputR;
};

class PseudorandomOverlappingFmpTicks : public FractionalMaxPoolingTicks {
public:
  PseudorandomOverlappingFmpTicks(int nIn, int nOut, int poolSize, RNG &rng);
};
class PseudorandomNonOverlappingFmpTicks : public FractionalMaxPoolingTicks {
public:
  PseudorandomNonOverlappingFmpTicks(int nIn, int nOut, int poolSize, RNG &rng);
};
class RandomOverlappingFmpTicks : public FractionalMaxPoolingTicks {
public:
  RandomOverlappingFmpTicks(int nIn, int nOut, int poolSize, RNG &rng);
};
class RandomNonOverlappingFmpTicks : public FractionalMaxPoolingTicks {
public:
  RandomNonOverlappingFmpTicks(int nIn, int nOut, int poolSize, RNG &rng);
};

template <typename ticks>
class FractionalPoolingRegions : public RectangularRegions {
  std::vector<ticks> pb;

public:
  FractionalPoolingRegions(int nIn, int nOut, int dimension, int poolSize,
                           RNG &rng);
  ~FractionalPoolingRegions();
  int inputL(int axis, int j);
  int inputR(int axis, int j);
  int outputL(int axis, int i);
  int outputR(int axis, int i);
};

void gridRules(SparseGrid &inputGrid, SparseGrid &outputGrid,
               RectangularRegions &regions, int &nOutputSpatialSites,
               std::vector<int> &rules, bool uniformSizeRegions,
               int minActiveInputs = 1);

class RegularTriangularRegions {
public:
  int poolSize;
  int poolStride;
  int nIn;  //  nIn==1: o    nIn==2: oo   nIn==3:  ooo   nIn==4: oooo   etc
            //  (o=site, x=nothing)
  int nOut; //                       ox            oox           ooox
  //                                                      oxx           ooxx
  //                                                                    oxxx
  // Similarly for nOut.
  int dimension; // dimension==2 for triangles, 3 for tetrahedron/pyramids, 4
                 // for tetrahedral hyperpyramids
  int s;         // 1d pooling region size (same as poolSize in
                 // RegularTriangularRegions)
  int S;         // total pooling region size (dimension==1, S==s; dimension==2,
                 // S=s*(s+1)/2, ....)
  std::vector<int> ord; // order of the S points chosen from the
                        // ipow(s,dimension) points in making up the filter
                        // shape
  RegularTriangularRegions(int nIn, int nOut, int dimension, int poolSize,
                           int poolStride);
  // up to 4d
  // given a point in the output layer, location of "top-left" corner of the
  // corresponding pooling region in the input layer
  int inputL(int j);
  // given an input-layer point, what is the range of points in the output layer
  // to which it can be pooled ((left<=x<right))
  int outputL(int i);
  int outputR(int i);
};

void gridRules(SparseGrid &inputGrid, // Keys 0,1,...,powf(regions.nIn,3)-1
                                      // represent grid points (plus paddding to
                                      // form a square/cube)
               SparseGrid &outputGrid, // Keys 0,1,...,powf(regions.nOut,3)-1
                                       // represent grid points (plus paddding
                                       // to form a square/cube)
               RegularTriangularRegions &regions, int &nOutputSpatialSites,
               std::vector<int> &rules, int minActiveInputs = 1);
