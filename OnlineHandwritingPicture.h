#pragma once

#include "Picture.h"
#include "Rng.h"
#include <armadillo>
#include <vector>

class OnlineHandwritingPicture : public Picture {
  float penSpeed3d;
  float offset3d;

public:
  std::vector<arma::mat> ops;
  int renderSize;
  OnlineHandwritingEncoding enc;
  OnlineHandwritingPicture(int renderSize, OnlineHandwritingEncoding enc,
                           int label, float penSpeed3d = 0.02);
  ~OnlineHandwritingPicture();
  void normalize(); // Fit centrally in the square
                    // [-renderSize/2,renderSize/2]^2
  Picture *distort(RNG &rng, batchType type);
  void codifyInputData(SparseGrid &grid, std::vector<float> &features,
                       int &nSpatialSites, int spatialSize);
  void jiggle(RNG &rng, float alpha);
  void draw(int spatialSize);
};
