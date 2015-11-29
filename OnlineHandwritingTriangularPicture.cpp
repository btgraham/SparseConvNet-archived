#include "OnlineHandwritingPicture.h"
#include "signature.h"

OnlineHandwritingPicture::OnlineHandwritingPicture(
    int renderSize, OnlineHandwritingEncoding enc, int label, float penSpeed3d)
    : enc(enc), renderSize(renderSize), Picture(label), offset3d(0),
      penSpeed3d(penSpeed3d) {}
OnlineHandwritingPicture::~OnlineHandwritingPicture() {}
void OnlineHandwritingPicture::normalize() { // Fit centrally in the cube
                                             // [-renderSize/2,renderSize/2]^2
  arma::mat pointsm(ops.size(), 2);
  arma::mat pointsM(ops.size(), 2);
  for (int i = 0; i < ops.size(); ++i) {
    pointsm.row(i) = arma::min(ops[i], 0);
    pointsM.row(i) = arma::max(ops[i], 0);
  }
  pointsm = arma::min(pointsm, 0);
  pointsM = arma::max(pointsM, 0);
  float scale = arma::mat(pointsM - pointsm).max();
  assert(scale > 0);
  for (int i = 0; i < ops.size(); ++i) {
    ops[i] = ops[i] - arma::repmat(0.5 * (pointsm + pointsM), ops[i].n_rows, 1);
    ops[i] *= renderSize / scale;
  }
}

arma::mat constantSpeed(arma::mat &stroke, float density, int multiplier = 1) {
  std::vector<float> lengths(stroke.n_rows);
  lengths[0] = 0;
  for (int i = 1; i < stroke.n_rows; i++) {
    lengths[i] =
        lengths[i - 1] + pow(pow(stroke(i, 0) - stroke(i - 1, 0), 2) +
                                 pow(stroke(i, 1) - stroke(i - 1, 1), 2),
                             0.5);
  }
  float lTotal = lengths[stroke.n_rows - 1];
  int n = (int)(0.5 + lTotal / density);
  n *= multiplier;
  arma::mat r(n + 1, 2);
  int j = 0;
  float alpha;
  r(0, 0) = stroke(0, 0);
  r(0, 1) = stroke(0, 1);
  for (int i = 1; i <= n; i++) {
    while (n * lengths[j + 1] < i * lTotal)
      j++;
    alpha = (lengths[j + 1] - i * lTotal / n) / (lengths[j + 1] - lengths[j]);
    r(i, 0) = stroke(j, 0) * alpha + stroke(j + 1, 0) * (1 - alpha);
    r(i, 1) = stroke(j, 1) * alpha + stroke(j + 1, 1) * (1 - alpha);
  }
  return r;
}

int maxLength = 0;

void OnlineHandwritingPicture::codifyInputData(SparseGrid &grid,
                                               std::vector<float> &features,
                                               int &nSpatialSites,
                                               int spatialSize) {
  int nInputFeatures = OnlineHandwritingEncodingSize[enc];
  // Background feature
  grid.backgroundCol = nSpatialSites++;
  features.resize(nInputFeatures * nSpatialSites, 0);
  // Render character
  switch (enc) {
  case Simple:
    for (int i = 0; i < ops.size(); i++) {
      arma::mat csp = constantSpeed(ops[i], 0.1);
      for (int j = 0; j < csp.n_rows; ++j) {
        int a0 = csp(j, 0) + spatialSize / 3.0,
            a1 = -0.57735 * csp(j, 0) + 1.1547 * csp(j, 1) + spatialSize / 3.0;
        if (a0 >= 0 and a1 >= 0 and a0 + a1 < spatialSize) {
          int n = a0 * spatialSize + a1;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          ++features[grid.mp[n]];
        }
      }
    }
    break;
  case Octogram: { // todo: Test
    float piBy4 = 3.1415926536 / 4;
    for (int i = 0; i < ops.size(); i++) {
      arma::mat csp = constantSpeed(ops[i], 0.1);
      for (int j = 0; j < csp.n_rows - 1; ++j) {
        int a0 = csp(j, 0) + spatialSize / 3.0,
            a1 = -0.57735 * csp(j, 0) + 1.1547 * csp(j, 1) + spatialSize / 3.0;
        if (a0 >= 0 and a1 >= 0 and a0 + a1 < spatialSize) {
          int n = a0 * spatialSize + a1;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          int mpnnIF = nInputFeatures * grid.mp[n];
          features[mpnnIF]++;
          float dx = (csp(j + 1, 0) - csp(j, 0));
          float dy = (csp(j + 1, 1) - csp(j, 1));
          float m = powf(dx * dx + dy * dy, 0.5);
          dx /= m;
          dy /= m;
          for (int k = 0; k < 8; k++) {
            float alpha =
                1 - acosf(cosf(k * piBy4) * dx + sinf(k * piBy4) * dy) / piBy4;
            if (alpha > 0)
              features[mpnnIF + k + 1] += alpha;
          }
        }
      }
    }
    break;
  }
  case LogSignature1:
    // Todo
    break;
  case LogSignature2:
    // Todo
    break;
  case LogSignature3:
    // Todo
    break;
  case LogSignature4:
    // Todo
    break;
  case SpaceTime3d: {
    std::vector<arma::mat> csp;
    int cspLengths = 0;
    for (int i = 0; i < ops.size(); i++) {
      csp.push_back(constantSpeed(ops[i], 0.1));
      cspLengths += csp[i].n_rows;
    }
    // // if (cspLengths>maxLength) {
    // //   std::cout <<cspLengths*penSpeed3d << " " << std::flush;
    // //   maxLength=cspLengths;
    // // }
    int z = 0;
    for (int i = 0; i < ops.size(); i++) {
      for (int j = 0; j < csp[i].n_rows; ++j) {
        int a0 = csp[i](j, 0) + spatialSize / 4.0,
            a1 = -0.57735 * csp[i](j, 0) + 1.1547 * csp[i](j, 1) +
                 spatialSize / 4.0,
            a2 = -0.40825 * csp[i](j, 0) - 0.40825 * csp[i](j, 1) +
                 1.2247 * renderSize *
                     ((z - cspLengths / 2) * penSpeed3d + offset3d) +
                 spatialSize / 4.0;
        z++;
        if (a0 >= 0 && a1 >= 0 && a2 >= 0 && a0 + a1 + a2 < spatialSize) {
          int n = a0 * spatialSize * spatialSize + a1 * spatialSize + a2;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          ++features[grid.mp[n]];
          /// std::cout << "."<<std::flush;
        } else
          std::cout << "x" << std::flush;
      }
    }
    break;
  }
  case VectorSpaceTime3d: {
    std::vector<arma::mat> csp;
    int cspLengths = 0;
    for (int i = 0; i < ops.size(); i++) {
      csp.push_back(constantSpeed(ops[i], 0.1));
      cspLengths += csp[i].n_rows;
    }
    if (cspLengths > maxLength) {
      std::cout << cspLengths *penSpeed3d << " " << std::flush;
      maxLength = cspLengths;
    }
    int z = 0;
    for (int i = 0; i < ops.size(); i++) {
      for (int j = 0; j < csp[i].n_rows - 1; ++j) {
        int a0 = csp[i](j, 0) + spatialSize / 4.0,
            a1 = -0.57735 * csp[i](j, 0) + 1.1547 * csp[i](j, 1) +
                 spatialSize / 4.0,
            a2 = -0.40825 * csp[i](j, 0) - 0.40825 * csp[i](j, 1) +
                 1.2247 * renderSize *
                     ((z - cspLengths / 2) * penSpeed3d + offset3d) +
                 spatialSize / 4.0;
        z++;
        if (a0 >= 0 && a1 >= 0 && a2 >= 0 && a0 + a1 + a2 < spatialSize) {
          int n = a0 * spatialSize * spatialSize + a1 * spatialSize + a2;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          int mpnnIF = nInputFeatures * grid.mp[n];
          features[mpnnIF]++;
          float dx = (csp[i](j + 1, 0) - csp[i](j, 0));
          float dy = (csp[i](j + 1, 1) - csp[i](j, 1));
          float m = powf(dx * dx + dy * dy, 0.5);
          dx /= m;
          dy /= m;
          features[mpnnIF + 1] += dx;
          features[mpnnIF + 2] += dy;
          //   std::cout << "."<<std::flush;
        } // else std::cout << "x"<<std::flush;
      }
    }
    break;
  }
  }
}

// Map tetrahedron to simplex
// arma::mat convPyrCub="1 0 0;  -0.57735   1.1547  0;  -0.40825  -0.40825
// 1.2247";
// (0,0,0) -> (0.0.0)
// (1,0,0) -> (1,0,0)
// (0.5,0.866,0) -> (0,1,0)
// (0.5,0.289,0.82) -> (0,0,1)

Picture *OnlineHandwritingPicture::distort(RNG &rng, batchType type) {
  OnlineHandwritingPicture *pic = new OnlineHandwritingPicture(*this);
  if (type == TRAINBATCH) {
    arma::mat aff = arma::eye(2, 2);
    aff(0, 0) += rng.uniform(-0.3, 0.3) / 1; // x stretch
    aff(1, 1) += rng.uniform(-0.3, 0.3) / 1; // y stretch
    int r = rng.randint(3);
    float alpha = rng.uniform(-0.3, 0.3) / 1;
    arma::mat x = arma::eye(2, 2);
    if (r == 0)
      x(0, 1) = alpha;
    if (r == 1)
      x(1, 0) = alpha;
    if (r = 2) {
      x(0, 0) = cos(alpha);
      x(0, 1) = -sin(alpha);
      x(1, 0) = sin(alpha);
      x(1, 1) = cos(alpha);
    }
    aff = aff * x;
    arma::mat y(1, 2);
    y(0, 0) = renderSize * rng.uniform(-0.1875, 0.1875) / 1;
    y(0, 1) = renderSize * rng.uniform(-0.1875, 0.1875) / 1;
    for (int i = 0; i < ops.size(); ++i)
      pic->ops[i] = pic->ops[i] * aff + arma::repmat(y, pic->ops[i].n_rows, 1);
    pic->offset3d = rng.uniform(-0.1, 0.1);
  }
  return pic;
}
