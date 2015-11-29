#include "Off3DFormatPicture.h"
#include <string>
#include <fstream>
// Draw triangle with vertices {a, a+u, a+v} on grid
// For every entry in grid (except the null vector -1) add a +1 to features
// vector.
void drawTriangleOFF(SparseGrid &grid, int inputFieldSize,
                     std::vector<float> &features, int &nSpatialSites, float a0,
                     float a1, float a2, float u0, float u1, float u2, float v0,
                     float v1, float v2) {
  float base = powf(u0 * u0 + u1 * u1 + u2 * u2, 0.5);
  u0 /= base;
  u1 /= base;
  u2 /= base;                                 // scale u to a unit vector
  float offset = u0 * v0 + u1 * v1 + u2 * v2; // u dot v
  v0 -= offset * u0;
  v1 -= offset * u1;
  v2 -= offset * u2; // make v orthogonal to u
  float height = powf(v0 * v0 + v1 * v1 + v2 * v2, 0.5);
  v0 /= height;
  v1 /= height;
  v2 /= height; // scale v to be a unit vector
  // u and v are now orthogonal
  // The triangle now has points {a, a+base*u, a+offset*u+height*v}

  for (float h = 0; h <= height; h = std::min(h + 1, height) + (h == height)) {
    float l = base * (1 - h / height);
    for (float b = 0; b <= l; b = std::min(b + 1, l) + (b == l)) {
      float p0 = a0 + (b + offset * h / height) * u0 + h * v0,
            p1 = a1 + (b + offset * h / height) * u1 + h * v1,
            p2 = a2 + (b + offset * h / height) * u2 + h * v2;
      int i0 = p0 + 0.25 * inputFieldSize, //(0.25,0.25,0.25) for the pyramid
          // center in cubic coord system,
          // corresponds to
          //(1/2,1/2/sqrt(3),1/sqrt(24))
          i1 = p1 + 0.25 * inputFieldSize, i2 = p2 + 0.25 * inputFieldSize;
      if (i0 >= 0 && i1 >= 0 && i2 >= 0 && i0 + i1 + i2 < inputFieldSize) {
        int n = i0 * inputFieldSize * inputFieldSize + i1 * inputFieldSize + i2;
        if (grid.mp.find(n) == grid.mp.end()) {
          grid.mp[n] = nSpatialSites++;
          features.push_back(1);
        }
      }
      // else
      //   std::cout << "!"<<std::flush;
    }
  }
}

OffSurfaceModelPicture::OffSurfaceModelPicture(std::string filename,
                                               int renderSize, int label)
    : Picture(label), renderSize(renderSize) {
  std::ifstream file(filename.c_str());
  std::string off;
  getline(file, off);
  int nPoints, nTriangles, nLines;
  file >> nPoints >> nTriangles >> nLines;
  points.set_size(nPoints, 3);
  surfaces.resize(nTriangles);
  for (int i = 0; i < nPoints; i++)
    file >> points(i, 0) >> points(i, 1) >> points(i, 2);
  for (int i = 0; i < nTriangles; i++) {
    surfaces[i].resize(3);
    int three;
    file >> three >> surfaces[i][0] >> surfaces[i][1] >> surfaces[i][2];
  }
}

OffSurfaceModelPicture::~OffSurfaceModelPicture() {}
void OffSurfaceModelPicture::normalize() { // Fit centrally in the cube
                                           // [-renderSize/2,renderSize/2]^3
  arma::mat pointsm = arma::min(points, 0);
  arma::mat pointsM = arma::max(points, 0);
  float scale = arma::mat(pointsM - pointsm).max();
  assert(scale > 0);
  points = points - arma::repmat(0.5 * (pointsm + pointsM), points.n_rows, 1);
  points *= renderSize / scale;
}

void OffSurfaceModelPicture::random_rotation(RNG &rng) {
  arma::mat L, Q, R;
  L.set_size(3, 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      L(i, j) = rng.uniform();
  arma::qr(Q, R, L);
  points = points * Q;
}

void OffSurfaceModelPicture::jiggle(RNG &rng, float alpha) {
  for (int i = 0; i < 3; i++)
    points.col(i) += renderSize * rng.uniform(-alpha, alpha);
}

void OffSurfaceModelPicture::affineTransform(RNG &rng, float alpha) {
  arma::mat L = arma::eye<arma::mat>(3, 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      L(i, j) += rng.uniform(-alpha, alpha);
  points = points * L;
}

void OffSurfaceModelPicture::codifyInputData(SparseGrid &grid,
                                             std::vector<float> &features,
                                             int &nSpatialSites,
                                             int spatialSize) {
  features.push_back(0); // Background feature
  grid.backgroundCol = nSpatialSites++;
  for (int i = 0; i < surfaces.size(); ++i) {
    // assume triangles
    drawTriangleOFF(grid, spatialSize, features, nSpatialSites,
                    points(surfaces[i][0], 0), points(surfaces[i][0], 1),
                    points(surfaces[i][0], 2),
                    points(surfaces[i][1], 0) - points(surfaces[i][0], 0),
                    points(surfaces[i][1], 1) - points(surfaces[i][0], 1),
                    points(surfaces[i][1], 2) - points(surfaces[i][0], 2),
                    points(surfaces[i][2], 0) - points(surfaces[i][0], 0),
                    points(surfaces[i][2], 1) - points(surfaces[i][0], 1),
                    points(surfaces[i][2], 2) - points(surfaces[i][0], 2));
  }
}

// Map tetrahedron to simplex
arma::mat convPyrCub =
    "1 0 0;  -0.57735   1.1547  0;  -0.40825  -0.40825 1.2247";
// (0,0,0) -> (0.0.0)
// (1,0,0) -> (1,0,0)
// (0.5,0.866,0) -> (0,1,0)
// (0.5,0.289,0.82) -> (0,0,10

Picture *OffSurfaceModelPicture::distort(RNG &rng, batchType type) {
  OffSurfaceModelPicture *pic = new OffSurfaceModelPicture(*this);
  pic->random_rotation(rng);
  pic->normalize();
  if (type == TRAINBATCH) {
    pic->affineTransform(rng, 0.2);
    pic->jiggle(rng, 0.2);
  }
  pic->points = pic->points * convPyrCub;
  return pic;
}
