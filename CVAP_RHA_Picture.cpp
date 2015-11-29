#include "CVAP_RHA_Picture.h"
#include <string>
#define WINDOW 100

CVAP_RHA_Picture::CVAP_RHA_Picture(std::ifstream &file)
    : xOffset(0), yOffset(0), tOffset(0), xVelocity(0), yVelocity(0) {
  std::string name;
  file >> name;
  if (name.empty())
    throw 1;
  file >> label;
  file >> nPoints;
  assert(nPoints > 0);
  data.resize(nPoints * 4);
  for (int i = 0; i < data.size(); ++i)
    file >> data[i];
  timescale = data[data.size() - 2];
  std::cout << label << " " << std::flush;
}

CVAP_RHA_Picture::~CVAP_RHA_Picture() {}

void CVAP_RHA_Picture::jiggle(RNG &rng, float alpha) {
  xOffset = rng.uniform(-alpha, alpha) * 120;
  yOffset = rng.uniform(-alpha, alpha) * 160;
  xVelocity = rng.uniform(-alpha, alpha);
  yVelocity = rng.uniform(-alpha, alpha);
  if (timescale < WINDOW)
    tOffset = rng.uniform(-WINDOW / 5, WINDOW / 5);
  else
    tOffset = rng.uniform(-1, 1) * (timescale - WINDOW) / 2;
}

void CVAP_RHA_Picture::affineTransform(RNG &rng, float alpha) {
  float a = rng.uniform(-alpha, alpha);
  float b = rng.uniform(-alpha, alpha);
  for (int i = 0; i < nPoints; i++) {
    data[4 * i + 0] += round(a * (data[4 * i + 1] - 80));
    data[4 * i + 1] += round(b * (data[4 * i + 0] - 60));
  }
}

void CVAP_RHA_Picture::codifyInputData(SparseGrid &grid,
                                       std::vector<float> &features,
                                       int &nSpatialSites, int spatialSize) {
  features.push_back(0); // Background feature
  grid.backgroundCol = nSpatialSites++;
  for (int i = 0; i < data.size(); i += 4) {
    int p0 = (data[i + 0] - 60) + xOffset +
             (int)(xVelocity * (data[i + 2] - timescale / 2 + tOffset)) +
             spatialSize / 2,
        p1 = (data[i + 1] - 80) + yOffset +
             (int)(yVelocity * (data[i + 2] - timescale / 2 + tOffset)) +
             spatialSize / 2,
        p2 = (data[i + 2] - timescale / 2 + tOffset) + spatialSize / 2;
    if (p0 >= 0 and p1 >= 0 and p2 >= 0 and p0 < spatialSize and
        p1 < spatialSize and p2 < spatialSize and
        abs(p2 - spatialSize / 2) < WINDOW / 2) {
      int n = p0 * spatialSize * spatialSize + p1 * spatialSize + p2;
      assert(grid.mp.find(n) == grid.mp.end());
      grid.mp[n] = nSpatialSites++;
      features.push_back(data[i + 3] / 128.0);
    }
  }
}

Picture *CVAP_RHA_Picture::distort(RNG &rng, batchType type) {
  CVAP_RHA_Picture *pic = new CVAP_RHA_Picture(*this);
  //  if (type==TRAINBATCH)
  pic->affineTransform(rng, 0.2);
  pic->jiggle(rng, 0.2);
  if (type == TESTBATCH)
    pic->xVelocity = pic->yVelocity = 0;
  return pic;
}
