#pragma once
#include "Picture.h"
#include "Rng.h"
#include <vector>
#include <fstream>

class CVAP_RHA_Picture : public Picture {
  int nPoints;

private:
  std::vector<short int> data;
  int timescale;
  int xOffset;
  int yOffset;
  int tOffset;
  float xVelocity;
  float yVelocity;

public:
  CVAP_RHA_Picture(std::ifstream &file);
  ~CVAP_RHA_Picture();
  void jiggle(RNG &rng, float alpha);
  void affineTransform(RNG &rng, float alpha);
  void codifyInputData(SparseGrid &grid, std::vector<float> &features,
                       int &nSpatialSites, int spatialSize);
  Picture *distort(RNG &rng, batchType type = TRAINBATCH);
};
