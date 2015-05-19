#include "Regions.h"
#include "utilities.h"

PoolingRegions::PoolingRegions(int nIn, int nOut, int dimension, int s)
  : nIn(nIn), nOut(nOut), dimension(dimension), s(s) {
  sd=ipow(s,dimension);
}
int PoolingRegions::tl0(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegions::tl1(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegions::tl2(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegions::tl3(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegions::lb0(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegions::ub0(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegions::lb1(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegions::ub1(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegions::lb2(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegions::ub2(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegions::lb3(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegions::ub3(int i0, int i1, int i2, int i3) {return 0;};


RegularPoolingRegions::RegularPoolingRegions(int nIn, int nOut, int dimension, int poolSize, int poolStride) :
  PoolingRegions(nIn,nOut,dimension, poolSize), poolSize(poolSize), poolStride(poolStride) {
  assert(nIn==poolSize+(nOut-1)*poolStride);
}
int RegularPoolingRegions::tl0(int j0, int j1, int j2, int j3) {return j0*poolStride;}
int RegularPoolingRegions::tl1(int j0, int j1, int j2, int j3) {return j1*poolStride;}
int RegularPoolingRegions::tl2(int j0, int j1, int j2, int j3) {return j2*poolStride;}
int RegularPoolingRegions::tl3(int j0, int j1, int j2, int j3) {return j3*poolStride;}
int RegularPoolingRegions::lb0(int i0, int i1, int i2, int i3) {return std::max(0,(i0-poolSize+poolStride)/poolStride);}
int RegularPoolingRegions::ub0(int i0, int i1, int i2, int i3) {return std::min(i0/poolStride,nOut-1);}
int RegularPoolingRegions::lb1(int i0, int i1, int i2, int i3) {return std::max(0,(i1-poolSize+poolStride)/poolStride);}
int RegularPoolingRegions::ub1(int i0, int i1, int i2, int i3) {return std::min(i1/poolStride,nOut-1);}
int RegularPoolingRegions::lb2(int i0, int i1, int i2, int i3) {return std::max(0,(i2-poolSize+poolStride)/poolStride);}
int RegularPoolingRegions::ub2(int i0, int i1, int i2, int i3) {return std::min(i2/poolStride,nOut-1);}
int RegularPoolingRegions::lb3(int i0, int i1, int i2, int i3) {return std::max(0,(i3-poolSize+poolStride)/poolStride);}
int RegularPoolingRegions::ub3(int i0, int i1, int i2, int i3) {return std::min(i3/poolStride,nOut-1);}


PseudorandomOverlappingFractionalMaxPoolingBlocks::PseudorandomOverlappingFractionalMaxPoolingBlocks(int nIn, int nOut, int poolSize, RNG& rng) {
  assert(nIn>=nOut-1+poolSize);
  float alpha=(nIn-poolSize)*1.0/(nOut-1);
  float u=rng.uniform(0,10000);
  /////////////////////////////////////////////////////////////////////////////////////////////////
  //Rougly speaking, we want to do this
  //   for (int i=0;i<nOut;++i)
  //     tl.push_back((int)((i+u)*alpha) - (int)(u*alpha));
  //After doing that, you might expect tl.back()==nIn-poolSize.
  //However, due to rounding effects, you sometimes get tl.back()=nIn-poolSize+1.
  //Therefore we do:
  for (int i=0;i<nOut-1;++i) //Iterate nOut-1 times ...
    tl.push_back((int)((i+u)*alpha) - (int)(u*alpha));
  tl.push_back(nIn-poolSize);// then add an nOut-th term almost corresponding to the case i=nOut-1
  /////////////////////////////////////////////////////////////////////////////////////////////////
  lb.resize(nIn,nOut);
  ub.resize(nIn,0);
  for (int i=0;i<nOut;i++) {
    for (int j=tl[i];j<tl[i]+poolSize;j++) {
      lb[j]=std::min(lb[j],i);
      ub[j]=std::max(ub[j],i);
    }
  }
}
PseudorandomOverlappingFractionalPoolingRegions::PseudorandomOverlappingFractionalPoolingRegions
(int nIn, int nOut, int dimension, int poolSize, RNG& rng) :
  PoolingRegions(nIn,nOut,dimension, poolSize) {
  for (int i=0;i<dimension;++i)
    pb.push_back(PseudorandomOverlappingFractionalMaxPoolingBlocks(nIn,nOut,poolSize,rng));
  assert(nIn>=nOut+poolSize-1);
}
int PseudorandomOverlappingFractionalPoolingRegions::tl0(int j0, int j1, int j2, int j3) {return pb[0].tl[j0];}
int PseudorandomOverlappingFractionalPoolingRegions::tl1(int j0, int j1, int j2, int j3) {return pb[1].tl[j1];}
int PseudorandomOverlappingFractionalPoolingRegions::tl2(int j0, int j1, int j2, int j3) {return pb[2].tl[j2];}
int PseudorandomOverlappingFractionalPoolingRegions::tl3(int j0, int j1, int j2, int j3) {return pb[3].tl[j3];}
int PseudorandomOverlappingFractionalPoolingRegions::lb0(int i0, int i1, int i2, int i3) {return pb[0].lb[i0];}
int PseudorandomOverlappingFractionalPoolingRegions::ub0(int i0, int i1, int i2, int i3) {return pb[0].ub[i0];}
int PseudorandomOverlappingFractionalPoolingRegions::lb1(int i0, int i1, int i2, int i3) {return pb[1].lb[i1];}
int PseudorandomOverlappingFractionalPoolingRegions::ub1(int i0, int i1, int i2, int i3) {return pb[1].ub[i1];}
int PseudorandomOverlappingFractionalPoolingRegions::lb2(int i0, int i1, int i2, int i3) {return pb[2].lb[i2];}
int PseudorandomOverlappingFractionalPoolingRegions::ub2(int i0, int i1, int i2, int i3) {return pb[2].ub[i2];}
int PseudorandomOverlappingFractionalPoolingRegions::lb3(int i0, int i1, int i2, int i3) {return pb[3].lb[i3];}
int PseudorandomOverlappingFractionalPoolingRegions::ub3(int i0, int i1, int i2, int i3) {return pb[3].ub[i3];}


RandomOverlappingFractionalMaxPoolingBlocks::RandomOverlappingFractionalMaxPoolingBlocks(int nIn, int nOut, int poolSize, RNG& rng) {
  assert(nIn>nOut-1+poolSize);
  std::vector<int> inc;
  float alpha=(nIn-poolSize)*1.0/(nOut-1);
  for (int i=0;i<nOut-1;i++)
    tl.push_back((int)((i+1)*alpha) - (int)((i)*alpha));
  rng.vectorShuffle(inc);
  tl.push_back(poolSize);
  tl.resize(1,0);
  for (int i=0;i<nOut-1;i++)
    tl.push_back(tl.back()+inc[i]);
  assert(tl.back()==nIn-poolSize);
  lb.resize(nIn,nOut);
  ub.resize(nIn,0);
  for (int i=0;i<nOut;i++) {
    for (int j=tl[i];j<tl[i]+poolSize;j++) {
      lb[j]=std::min(lb[j],i);
      ub[j]=std::max(ub[j],i);
    }
  }
}
RandomOverlappingFractionalPoolingRegions::RandomOverlappingFractionalPoolingRegions
(int nIn, int nOut, int dimension, int poolSize, RNG& rng) :
  PoolingRegions(nIn,nOut,dimension, poolSize) {
  for (int i=0;i<dimension;++i)
    pb.push_back(RandomOverlappingFractionalMaxPoolingBlocks(nIn,nOut,poolSize,rng));
  assert(nIn>nOut+poolSize-1);
}
int RandomOverlappingFractionalPoolingRegions::tl0(int j0, int j1, int j2, int j3) {return pb[0].tl[j0];}
int RandomOverlappingFractionalPoolingRegions::tl1(int j0, int j1, int j2, int j3) {return pb[1].tl[j1];}
int RandomOverlappingFractionalPoolingRegions::tl2(int j0, int j1, int j2, int j3) {return pb[2].tl[j2];}
int RandomOverlappingFractionalPoolingRegions::tl3(int j0, int j1, int j2, int j3) {return pb[3].tl[j3];}
int RandomOverlappingFractionalPoolingRegions::lb0(int i0, int i1, int i2, int i3) {return pb[0].lb[i0];}
int RandomOverlappingFractionalPoolingRegions::ub0(int i0, int i1, int i2, int i3) {return pb[0].ub[i0];}
int RandomOverlappingFractionalPoolingRegions::lb1(int i0, int i1, int i2, int i3) {return pb[1].lb[i1];}
int RandomOverlappingFractionalPoolingRegions::ub1(int i0, int i1, int i2, int i3) {return pb[1].ub[i1];}
int RandomOverlappingFractionalPoolingRegions::lb2(int i0, int i1, int i2, int i3) {return pb[2].lb[i2];}
int RandomOverlappingFractionalPoolingRegions::ub2(int i0, int i1, int i2, int i3) {return pb[2].ub[i2];}
int RandomOverlappingFractionalPoolingRegions::lb3(int i0, int i1, int i2, int i3) {return pb[3].lb[i3];}
int RandomOverlappingFractionalPoolingRegions::ub3(int i0, int i1, int i2, int i3) {return pb[3].ub[i3];}


void gridRules
(SparseGrid &inputGrid, //Keys 0,1,...,powf(regions.nIn,dimension)-1 represent grid points
 SparseGrid &outputGrid, //Keys 0,1,...,powf(regions.nOut,dimension)-1 represent grid points
 PoolingRegions& regions,
 int& nOutputSpatialSites,
 std::vector<int>& rules) {
  switch(regions.dimension) {
  case 1:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first;
      for (int j0=regions.lb0(i0);j0<=regions.ub0(i0);++j0) {
        int64_t key=(int64_t)j0;
        if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
          for (int i=0;i<regions.sd;++i) rules.push_back(inputGrid.backgroundCol);
          outputGrid.mp[key]=nOutputSpatialSites++;
        }
        rules[outputGrid.mp[key]*regions.sd
              + (i0-regions.tl0(j0))]
          = iter->second;
      }
    }
    break;
  case 2:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first/regions.nIn;
      int i1=(iter->first)%regions.nIn;
      for (int j0=regions.lb0(i0,i1);j0<=regions.ub0(i0,i1);++j0) {
        for (int j1=regions.lb1(i0,i1);j1<=regions.ub1(i0,i1);++j1) {
          int64_t key=(int64_t)j0*regions.nOut + (int64_t)j1;
          if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
            for (int i=0;i<regions.sd;++i) rules.push_back(inputGrid.backgroundCol);
            outputGrid.mp[key]=nOutputSpatialSites++;
          }
          rules[outputGrid.mp[key]*regions.sd
                + (i0-regions.tl0(j0,j1))*regions.s
                + (i1-regions.tl1(j0,j1))]
            = iter->second;
        }
      }
    }
    break;
  case 3:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first/regions.nIn/regions.nIn;
      int i1=((iter->first)/regions.nIn)%regions.nIn;
      int i2=(iter->first)%regions.nIn;
      for (int j0=regions.lb0(i0,i1,i2);j0<=regions.ub0(i0,i1,i2);++j0) {
        for (int j1=regions.lb1(i0,i1,i2);j1<=regions.ub1(i0,i1,i2);++j1) {
          for (int j2=regions.lb2(i0,i1,i2);j2<=regions.ub2(i0,i1,i2);++j2) {
            int64_t key=(int64_t)j0*regions.nOut*regions.nOut + (int64_t)j1*regions.nOut + (int64_t)j2;
            if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
              for (int i=0;i<regions.sd;++i) rules.push_back(inputGrid.backgroundCol);
              outputGrid.mp[key]=nOutputSpatialSites++;
            }
            rules[outputGrid.mp[key]*regions.sd
                  + (i0-regions.tl0(j0,j1,j2))*regions.s*regions.s
                  + (i1-regions.tl1(j0,j1,j2))*regions.s
                  + (i2-regions.tl2(j0,j1,j2))]
              = iter->second;
          }
        }
      }
    }
    break;
  case 4:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first/regions.nIn/regions.nIn/regions.nIn;
      int i1=((iter->first)/regions.nIn/regions.nIn)%regions.nIn;
      int i2=((iter->first)/regions.nIn)%regions.nIn;
      int i3=(iter->first)%regions.nIn;
      for (int j0=regions.lb0(i0,i1,i2,i3);j0<=regions.ub0(i0,i1,i2,i3);++j0) {
        for (int j1=regions.lb1(i0,i1,i2,i3);j1<=regions.ub1(i0,i1,i2,i3);++j1) {
          for (int j2=regions.lb2(i0,i1,i2,i3);j2<=regions.ub2(i0,i1,i2,i3);++j2) {
            for (int j3=regions.lb3(i0,i1,i2,i3);j3<=regions.ub3(i0,i1,i2,i3);++j3) {
              int64_t key=(int64_t)j0*regions.nOut*regions.nOut*regions.nOut + (int64_t)j1*regions.nOut*regions.nOut + (int64_t)j2*regions.nOut + (int64_t)j3;
              if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
                for (int i=0;i<regions.sd;++i) rules.push_back(inputGrid.backgroundCol);
                outputGrid.mp[key]=nOutputSpatialSites++;
              }
              rules[outputGrid.mp[key]*regions.sd
                    + (i0-regions.tl0(j0,j1,j2,j3))*regions.s*regions.s*regions.s
                    + (i1-regions.tl1(j0,j1,j2,j3))*regions.s*regions.s
                    + (i2-regions.tl2(j0,j1,j2,j3))*regions.s
                    + (i3-regions.tl3(j0,j1,j2,j3))]
                = iter->second;
            }
          }
        }
      }
    }
    break;
  }
  if (outputGrid.mp.size()< ipow(regions.nOut,regions.dimension)) { //Null vector/background needed
    for (int i=0;i<regions.sd;++i) rules.push_back(inputGrid.backgroundCol);
    outputGrid.backgroundCol=nOutputSpatialSites++;
  }
}











PoolingRegionsTriangular::PoolingRegionsTriangular(int nIn, int nOut, int dimension, int s) : nIn(nIn), nOut(nOut), dimension(dimension), s(s) {
  S=0;    //Calculate #points in the triangular filter, and order them
  ord.resize(ipow(s,dimension),-1); //iterate over the s^d cube, -1 means not in the filter
  for (int i=0;i<ord.size();i++) {
    int j=i,J=0;
    while (j>0) {
      J+=j%s;  //Calulate the L1-norm (Taxicab norm) of the points location
      j/=s;
    }
    if (J<s) //if the points lies in the triangle/pyramid, add it to the filter
      ord[i]=S++;
  }
}

int PoolingRegionsTriangular::tl0(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegionsTriangular::tl1(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegionsTriangular::tl2(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegionsTriangular::tl3(int j0, int j1, int j2, int j3) {return 0;};
int PoolingRegionsTriangular::lb0(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegionsTriangular::ub0(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegionsTriangular::lb1(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegionsTriangular::ub1(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegionsTriangular::lb2(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegionsTriangular::ub2(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegionsTriangular::lb3(int i0, int i1, int i2, int i3) {return 0;};
int PoolingRegionsTriangular::ub3(int i0, int i1, int i2, int i3) {return 0;};

RegularPoolingRegionsTriangular::RegularPoolingRegionsTriangular(int nIn, int nOut, int dimension, int poolSize, int poolStride) : PoolingRegionsTriangular(nIn,nOut,dimension, poolSize), poolSize(poolSize), poolStride(poolStride) {
  assert(nIn==poolSize+(nOut-1)*poolStride);
}
int RegularPoolingRegionsTriangular::tl0(int j0, int j1, int j2, int j3) {return j0*poolStride;}
int RegularPoolingRegionsTriangular::tl1(int j0, int j1, int j2, int j3) {return j1*poolStride;}
int RegularPoolingRegionsTriangular::tl2(int j0, int j1, int j2, int j3) {return j2*poolStride;}
int RegularPoolingRegionsTriangular::tl3(int j0, int j1, int j2, int j3) {return j3*poolStride;}
int RegularPoolingRegionsTriangular::lb0(int i0, int i1, int i2, int i3) {return std::max(0,(i0-poolSize+poolStride)/poolStride);}
int RegularPoolingRegionsTriangular::ub0(int i0, int i1, int i2, int i3) {return std::min(i0/poolStride,nOut-1);}
int RegularPoolingRegionsTriangular::lb1(int i0, int i1, int i2, int i3) {return std::max(0,(i1-poolSize+poolStride)/poolStride);}
int RegularPoolingRegionsTriangular::ub1(int i0, int i1, int i2, int i3) {return std::min(i1/poolStride,nOut-1);}
int RegularPoolingRegionsTriangular::lb2(int i0, int i1, int i2, int i3) {return std::max(0,(i2-poolSize+poolStride)/poolStride);}
int RegularPoolingRegionsTriangular::ub2(int i0, int i1, int i2, int i3) {return std::min(i2/poolStride,nOut-1);}
int RegularPoolingRegionsTriangular::lb3(int i0, int i1, int i2, int i3) {return std::max(0,(i3-poolSize+poolStride)/poolStride);}
int RegularPoolingRegionsTriangular::ub3(int i0, int i1, int i2, int i3) {return std::min(i3/poolStride,nOut-1);}

void gridRulesTriangular
(SparseGrid& inputGrid, //Keys 0,1,...,powf(regions.nIn,3)-1 represent grid points (plus paddding to form a square/cube); key -1 represents null/background vector
 SparseGrid& outputGrid, //Keys 0,1,...,powf(regions.nOut,3)-1 represent grid points (plus paddding to form a square/cube); key -1 represents null/background vector
 PoolingRegionsTriangular& regions,
 int& nOutputSpatialSites,
 std::vector<int>& rules) {
  switch(regions.dimension) {
  case 1:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first;
      for (int j0=regions.lb0(i0);j0<=regions.ub0(i0);++j0) {
        int64_t key=(int64_t)j0;
        int r=regions.ord[ (i0-regions.tl0(j0)) ];
        if (r>=0) {
          if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
            for (int i=0;i<regions.S;++i) rules.push_back(inputGrid.backgroundCol);
            outputGrid.mp[key]=nOutputSpatialSites++;
          }
          rules[   outputGrid.mp[key]*regions.S   + r  ] = iter->second;
        }
      }
    }
    break;
  case 2:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first/regions.nIn;
      int i1=(iter->first)%regions.nIn;
      for (int j0=regions.lb0(i0,i1);j0<=regions.ub0(i0,i1);++j0) {
        for (int j1=regions.lb1(i0,i1);j1<=regions.ub1(i0,i1);++j1) {
          if (j0+j1<regions.nOut) {
            int64_t key=(int64_t)j0*regions.nOut + (int64_t)j1;
            int r=regions.ord[ (i0-regions.tl0(j0,j1))*regions.s +
                               (i1-regions.tl1(j0,j1))];
            if (r>=0) {
              if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
                for (int i=0;i<regions.S;++i) {
                  rules.push_back(inputGrid.backgroundCol);
                }
                outputGrid.mp[key]=nOutputSpatialSites++;
              }
              rules[ outputGrid.mp[key]*regions.S + r ] = iter->second;
            }
          }
        }
      }
    }
    break;
  case 3:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first/regions.nIn/regions.nIn;
      int i1=((iter->first)/regions.nIn)%regions.nIn;
      int i2=(iter->first)%regions.nIn;
      for (int j0=regions.lb0(i0,i1,i2);j0<=regions.ub0(i0,i1,i2);++j0) {
        for (int j1=regions.lb1(i0,i1,i2);j1<=regions.ub1(i0,i1,i2);++j1) {
          for (int j2=regions.lb2(i0,i1,i2);j2<=regions.ub2(i0,i1,i2);++j2) {
            if (j0+j1+j2<regions.nOut) {
              int64_t key=(int64_t)j0*regions.nOut*regions.nOut + (int64_t)j1*regions.nOut + (int64_t)j2;
              int r=regions.ord[ (i0-regions.tl0(j0,j1,j2))*regions.s*regions.s +
                                 (i1-regions.tl1(j0,j1,j2))*regions.s +
                                 (i2-regions.tl2(j0,j1,j2)) ];
              if (r>=0) {
                if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
                  for (int i=0;i<regions.S;++i) rules.push_back(inputGrid.backgroundCol);
                  outputGrid.mp[key]=nOutputSpatialSites++;
                }
                rules[   outputGrid.mp[key]*regions.S  + r  ] = iter->second;
              }
            }
          }
        }
      }
    }
    break;
  case 4:
    for (SparseGridIter iter = inputGrid.mp.begin();iter != inputGrid.mp.end(); ++iter) {
      int i0=iter->first/regions.nIn/regions.nIn/regions.nIn;
      int i1=((iter->first)/regions.nIn/regions.nIn)%regions.nIn;
      int i2=((iter->first)/regions.nIn)%regions.nIn;
      int i3=(iter->first)%regions.nIn;
      for (int j0=regions.lb0(i0,i1,i2,i3);j0<=regions.ub0(i0,i1,i2,i3);++j0) {
        for (int j1=regions.lb1(i0,i1,i2,i3);j1<=regions.ub1(i0,i1,i2,i3);++j1) {
          for (int j2=regions.lb2(i0,i1,i2,i3);j2<=regions.ub2(i0,i1,i2,i3);++j2) {
            for (int j3=regions.lb3(i0,i1,i2,i3);j3<=regions.ub3(i0,i1,i2,i3);++j3) {
              if (j0+j1+j2+j3<regions.nOut) {
                int64_t key=(int64_t)j0*regions.nOut*regions.nOut*regions.nOut + (int64_t)j1*regions.nOut*regions.nOut + (int64_t)j2*regions.nOut + (int64_t)j3;
                int r=regions.ord[(i0-regions.tl0(j0,j1,j2,j3))*regions.s*regions.s*regions.s +
                                  (i1-regions.tl1(j0,j1,j2,j3))*regions.s*regions.s +
                                  (i2-regions.tl2(j0,j1,j2,j3))*regions.s +
                                  (i3-regions.tl3(j0,j1,j2,j3)) ];
                if (r>=0) {
                  if(outputGrid.mp.find(key)==outputGrid.mp.end()) { // Add line to rules
                    for (int i=0;i<regions.S;++i) rules.push_back(inputGrid.backgroundCol);
                    outputGrid.mp[key]=nOutputSpatialSites++;
                  }
                  rules[   outputGrid.mp[key]*regions.S + r ] = iter->second;
                }
              }
            }
          }
        }
      }
    }
    break;
  }
  if (outputGrid.mp.size()< triangleSize(regions.nOut,regions.dimension)) { //Null vector/background needed
    for (int i=0;i<regions.S;++i) rules.push_back(inputGrid.backgroundCol);
    outputGrid.backgroundCol=nOutputSpatialSites++;
  }
}
