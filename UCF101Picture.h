#pragma once
#include "Picture.h"
#include "Rng.h"
#include <vector>
#include <fstream>

class UCF101Picture : public Picture {
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
  UCF101Picture(std::ifstream &file);
  ~UCF101Picture();
  void jiggle(RNG &rng, float alpha);
  void affineTransform(RNG &rng, float alpha);
  void codifyInputData(SparseGrid &grid, std::vector<float> &features,
                       int &nSpatialSites, int spatialSize);
  Picture *distort(RNG &rng, batchType type = TRAINBATCH);
};
