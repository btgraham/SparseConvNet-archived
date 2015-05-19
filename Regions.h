#pragma once
#include <vector>
#include "Rng.h"
#include "SparseGrid.h"


//All pooling regions are all of equal size.
//(i.e. We implement overlapping fractional max-pooling, but not non-overlapping fractional max-pooling)

class PoolingRegions { //Which output locations does the (i,j)-th input cell feed into (inclusive bounds)
public:
  int nIn;
  int nOut;
  int dimension;
  int s;
  int sd;
  PoolingRegions(int nIn, int nOut, int dimension, int s);
  //up to 4d
  //given a point in the output layer, location of "top-left" corner of the corresponding pooling region in the input layer
  virtual int tl0(int j0, int j1=0, int j2=0, int j3=0);
  virtual int tl1(int j0, int j1=0, int j2=0, int j3=0);
  virtual int tl2(int j0, int j1=0, int j2=0, int j3=0);
  virtual int tl3(int j0, int j1=0, int j2=0, int j3=0);
  //given an input-layer point, what is the range of points in the output layer to which can be pooled (inclusive bounds)
  virtual int lb0(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub0(int i0, int i1=0, int i2=0, int i3=0);
  virtual int lb1(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub1(int i0, int i1=0, int i2=0, int i3=0);
  virtual int lb2(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub2(int i0, int i1=0, int i2=0, int i3=0);
  virtual int lb3(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub3(int i0, int i1=0, int i2=0, int i3=0);
};

class RegularPoolingRegions : public PoolingRegions {
  int poolSize;
  int poolStride;
public:
  RegularPoolingRegions(int nIn, int nOut, int dimension, int poolSize, int poolStride);
  int tl0(int j0, int j1=0, int j2=0, int j3=0);
  int tl1(int j0, int j1=0, int j2=0, int j3=0);
  int tl2(int j0, int j1=0, int j2=0, int j3=0);
  int tl3(int j0, int j1=0, int j2=0, int j3=0);

  int lb0(int i0, int i1=0, int i2=0, int i3=0);
  int ub0(int i0, int i1=0, int i2=0, int i3=0);
  int lb1(int i0, int i1=0, int i2=0, int i3=0);
  int ub1(int i0, int i1=0, int i2=0, int i3=0);
  int lb2(int i0, int i1=0, int i2=0, int i3=0);
  int ub2(int i0, int i1=0, int i2=0, int i3=0);
  int lb3(int i0, int i1=0, int i2=0, int i3=0);
  int ub3(int i0, int i1=0, int i2=0, int i3=0);
};

class PseudorandomOverlappingFractionalMaxPoolingBlocks {
public:
  std::vector<int> tl;
  std::vector<int> lb;
  std::vector<int> ub;
  PseudorandomOverlappingFractionalMaxPoolingBlocks(int nIn, int nOut, int poolSize, RNG& rng);
};
class PseudorandomOverlappingFractionalPoolingRegions : public PoolingRegions { //For 2d pooling.
  std::vector<PseudorandomOverlappingFractionalMaxPoolingBlocks> pb;
public:
  PseudorandomOverlappingFractionalPoolingRegions(int nIn, int nOut, int dimension, int poolSize, RNG& rng);
  int tl0(int j0, int j1=0, int j2=0, int j3=0);
  int tl1(int j0, int j1=0, int j2=0, int j3=0);
  int tl2(int j0, int j1=0, int j2=0, int j3=0);
  int tl3(int j0, int j1=0, int j2=0, int j3=0);
  int lb0(int i0, int i1=0, int i2=0, int i3=0);
  int ub0(int i0, int i1=0, int i2=0, int i3=0);
  int lb1(int i0, int i1=0, int i2=0, int i3=0);
  int ub1(int i0, int i1=0, int i2=0, int i3=0);
  int lb2(int i0, int i1=0, int i2=0, int i3=0);
  int ub2(int i0, int i1=0, int i2=0, int i3=0);
  int lb3(int i0, int i1=0, int i2=0, int i3=0);
  int ub3(int i0, int i1=0, int i2=0, int i3=0);
};

class RandomOverlappingFractionalMaxPoolingBlocks {
public:
  std::vector<int> tl;
  std::vector<int> lb;
  std::vector<int> ub;
  RandomOverlappingFractionalMaxPoolingBlocks(int nIn, int nOut, int poolSize, RNG& rng);
};
class RandomOverlappingFractionalPoolingRegions : public PoolingRegions {
  std::vector<RandomOverlappingFractionalMaxPoolingBlocks> pb;
public:
  RandomOverlappingFractionalPoolingRegions(int nIn, int nOut, int dimension, int poolSize, RNG& rng);
  int tl0(int j0, int j1=0, int j2=0, int j3=0);
  int tl1(int j0, int j1=0, int j2=0, int j3=0);
  int tl2(int j0, int j1=0, int j2=0, int j3=0);
  int tl3(int j0, int j1=0, int j2=0, int j3=0);
  int lb0(int i0, int i1=0, int i2=0, int i3=0);
  int ub0(int i0, int i1=0, int i2=0, int i3=0);
  int lb1(int i0, int i1=0, int i2=0, int i3=0);
  int ub1(int i0, int i1=0, int i2=0, int i3=0);
  int lb2(int i0, int i1=0, int i2=0, int i3=0);
  int ub2(int i0, int i1=0, int i2=0, int i3=0);
  int lb3(int i0, int i1=0, int i2=0, int i3=0);
  int ub3(int i0, int i1=0, int i2=0, int i3=0);
};


void gridRules
(SparseGrid& inputGrid, //Keys 0,1,...,powf(regions.nIn,3)-1 represent grid points; key -1 represents null/background std::vector
 SparseGrid& outputGrid, //Keys 0,1,...,powf(regions.nOut,3)-1 represent grid points; key -1 represents null/background std::vector
 PoolingRegions& regions,
 int& nOutputSpatialSites,
 std::vector<int>& rules);





















class PoolingRegionsTriangular { //Which output locations does the (i,j)-th input cell feed into (inclusive bounds)
public:
  int nIn;         //  nIn==1: o    nIn==2: oo   nIn==3:  ooo   nIn==4: oooo   etc (o=site, x=nothing)
  int nOut;        //                       ox            oox           ooox
  //                                                      oxx           ooxx
  //                                                                    oxxx
  // Similarly for nOut.
  int dimension;   // dimension==2 for triangles, 3 for tetrahedron/pyramids, 4 for tetrahedral hyperpyramids
  int s;           // 1d pooling region size (same as poolSize in RegularPoolingRegionsTriangular)
  int S;           // total pooling region size (dimension==1, S==s; dimension==2, S=s*(s+1)/2, ....)
  std::vector<int> ord; // order of the S points chosen from the ipow(s,dimension) points in making up the filter shape
  PoolingRegionsTriangular(int nIn, int nOut, int dimension, int s);
  //up to 4d
  //given a point in the output layer, location of "top-left" corner of the corresponding pooling region in the input layer
  virtual int tl0(int j0, int j1=0, int j2=0, int j3=0);
  virtual int tl1(int j0, int j1=0, int j2=0, int j3=0);
  virtual int tl2(int j0, int j1=0, int j2=0, int j3=0);
  virtual int tl3(int j0, int j1=0, int j2=0, int j3=0);
  //given an input-layer point, what is the range of points in the output layer to which can be pooled (inclusive bounds)
  virtual int lb0(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub0(int i0, int i1=0, int i2=0, int i3=0);
  virtual int lb1(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub1(int i0, int i1=0, int i2=0, int i3=0);
  virtual int lb2(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub2(int i0, int i1=0, int i2=0, int i3=0);
  virtual int lb3(int i0, int i1=0, int i2=0, int i3=0);
  virtual int ub3(int i0, int i1=0, int i2=0, int i3=0);
};

class RegularPoolingRegionsTriangular : public PoolingRegionsTriangular {
  int poolSize;
  int poolStride;
public:
  RegularPoolingRegionsTriangular(int nIn, int nOut, int dimension, int poolSize, int poolStride);
  int tl0(int j0, int j1=0, int j2=0, int j3=0);
  int tl1(int j0, int j1=0, int j2=0, int j3=0);
  int tl2(int j0, int j1=0, int j2=0, int j3=0);
  int tl3(int j0, int j1=0, int j2=0, int j3=0);

  int lb0(int i0, int i1=0, int i2=0, int i3=0);
  int ub0(int i0, int i1=0, int i2=0, int i3=0);
  int lb1(int i0, int i1=0, int i2=0, int i3=0);
  int ub1(int i0, int i1=0, int i2=0, int i3=0);
  int lb2(int i0, int i1=0, int i2=0, int i3=0);
  int ub2(int i0, int i1=0, int i2=0, int i3=0);
  int lb3(int i0, int i1=0, int i2=0, int i3=0);
  int ub3(int i0, int i1=0, int i2=0, int i3=0);
};



void gridRulesTriangular
(SparseGrid& inputGrid, //Keys 0,1,...,powf(regions.nIn,3)-1 represent grid points (plus paddding to form a square/cube); key -1 represents null/background vector
 SparseGrid& outputGrid, //Keys 0,1,...,powf(regions.nOut,3)-1 represent grid points (plus paddding to form a square/cube); key -1 represents null/background vector
 PoolingRegionsTriangular& regions,
 int& nOutputSpatialSites,
 std::vector<int>& rules);
