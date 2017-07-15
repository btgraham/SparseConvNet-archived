#include "OnlineHandwritingPicture.h"
#include "signature.h"

OnlineHandwritingPicture::OnlineHandwritingPicture(
    int renderSize, OnlineHandwritingEncoding enc, int label, float penSpeed3d)
    : Picture(label), penSpeed3d(penSpeed3d), offset3d(0),
      renderSize(renderSize), enc(enc) {}
OnlineHandwritingPicture::~OnlineHandwritingPicture() {}
void OnlineHandwritingPicture::normalize() { // Fit centrally in the cube
                                             // [-renderSize/2,renderSize/2]^2
  arma::mat pointsm(ops.size(), 2);
  arma::mat pointsM(ops.size(), 2);
  for (unsigned int i = 0; i < ops.size(); ++i) {
    pointsm.row(i) = arma::min(ops[i], 0);
    pointsM.row(i) = arma::max(ops[i], 0);
  }
  pointsm = arma::min(pointsm, 0);
  pointsM = arma::max(pointsM, 0);
  float scale = arma::mat(pointsM - pointsm).max();
  assert(scale > 0);
  for (unsigned int i = 0; i < ops.size(); ++i) {
    ops[i] = ops[i] - arma::repmat(0.5 * (pointsm + pointsM), ops[i].n_rows, 1);
    ops[i] *= renderSize / scale;
  }
}

arma::mat constantSpeed(arma::mat &stroke, float density, int multiplier = 1) {
  std::vector<float> lengths(stroke.n_rows);
  lengths[0] = 0;
  for (unsigned int i = 1; i < stroke.n_rows; i++) {
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

std::vector<std::vector<float>>
signatureWindows(arma::mat &path, int logSigDepth, int windowSize) {
  std::vector<std::vector<float>> sW(path.n_rows);
  std::vector<float> repPath;
  for (unsigned int i = 0; i < path.n_rows; ++i) {
    repPath.push_back(i);
    repPath.push_back(path(i, 0));
    repPath.push_back(path(i, 1));
  }
  for (unsigned int i = 0; i < path.size(); ++i) {
    sW[i].resize(logsigdim(3, logSigDepth));
    int first = std::max(0, (int)i - windowSize);
    int last = std::min(path.size() - 1, i + windowSize);
    logSignature(&repPath[3 * first], last - first + 1, 3, logSigDepth,
                 &sW[i][0]);
  }
  return sW;
}

int mapToGrid(float coord, int scale_N) {
  return std::max(0, std::min(scale_N - 1, (int)(coord + 0.5 * scale_N)));
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
  int logSigDepth;
  // Render character
  switch (enc) {
  case Simple:
    for (unsigned int i = 0; i < ops.size(); i++) {
      arma::mat csp = constantSpeed(ops[i], 0.1);
      for (unsigned int j = 0; j < csp.n_rows; ++j) {
        int a0 = csp(j, 0) + spatialSize / 2.0,
            a1 = csp(j, 1) + spatialSize / 2.0;
        if (a0 >= 0 and a1 >= 0 and a0 < spatialSize and a1 < spatialSize) {
          int n = a0 * spatialSize + a1;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          ++features[grid.mp[n]];
          //          std::cout <<"."<<std::flush;
        } // else  std::cout <<"x"<<std::flush;
      }
    }
    break;
  case Octogram: {
    float piBy4 = 3.1415926536 / 4;
    for (unsigned int i = 0; i < ops.size(); i++) {
      arma::mat csp = constantSpeed(ops[i], 0.1);
      for (unsigned int j = 0; j < csp.n_rows - 1; ++j) {
        int a0 = csp(j, 0) + spatialSize / 2.0,
            a1 = csp(j, 1) + spatialSize / 2.0;
        if (a0 >= 0 and a1 >= 0 and a0 < spatialSize and a1 < spatialSize) {
          int n = a0 * spatialSize + a1;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          int mpnnIF = nInputFeatures * grid.mp[n];
          float dx = (csp(j + 1, 0) - csp(j, 0));
          float dy = (csp(j + 1, 1) - csp(j, 1));
          float m = powf(dx * dx + dy * dy, 0.5);
          dx /= m;
          dy /= m;
          for (int k = 0; k < 8; k++) {
            float alpha =
                1 - acosf(cosf(k * piBy4) * dx + sinf(k * piBy4) * dy) / piBy4;
            if (alpha > 0)
              features[mpnnIF + k] += alpha;
          }
        }
      }
    }
  } break;
  case UndirectedOctogram: {
    float piBy4 = 3.1415926536 / 4;
    for (unsigned int i = 0; i < ops.size(); i++) {
      arma::mat csp = constantSpeed(ops[i], 0.1);
      for (unsigned int j = 0; j < csp.n_rows - 1; ++j) {
        int a0 = csp(j, 0) + spatialSize / 2.0,
            a1 = csp(j, 1) + spatialSize / 2.0;
        if (a0 >= 0 and a1 >= 0 and a0 < spatialSize and a1 < spatialSize) {
          int n = a0 * spatialSize + a1;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          int mpnnIF = nInputFeatures * grid.mp[n];
          float dx = (csp(j + 1, 0) - csp(j, 0));
          float dy = (csp(j + 1, 1) - csp(j, 1));
          float m = powf(dx * dx + dy * dy, 0.5);
          dx /= m;
          dy /= m;
          for (int k = 0; k < 4; k++) {
            float alpha =
                1 -
                acosf(fabs(cosf(k * piBy4) * dx + sinf(k * piBy4) * dy)) /
                    piBy4;
            if (alpha > 0)
              features[mpnnIF + k] += alpha;
          }
        }
      }
    }
  } break;
  case LogSignature1:
    logSigDepth = 1; // todo: test
  case LogSignature2:
    logSigDepth = 2;
  case LogSignature3:
    logSigDepth = 3;
  case LogSignature4:
    logSigDepth = 4;
    {
      int windowSize = 4;
      float scale = renderSize / 5;
      int multiplier = std::max((int)(2 * scale + 0.5), 1);
      // Todo
      for (unsigned int i = 0; i < ops.size(); i++) {
        arma::mat CSP = constantSpeed(ops[i], scale, 1);
        arma::mat csp = constantSpeed(ops[i], scale, multiplier);
        std::vector<std::vector<float>> sW =
            signatureWindows(CSP, logSigDepth, windowSize);
        for (unsigned int j = 0; j < csp.n_rows; ++j) {
          int J = j / multiplier;
          int a0 = csp(j, 0) + spatialSize / 2.0,
              a1 = csp(j, 1) + spatialSize / 2.0;
          if (a0 >= 0 and a1 >= 0 and a0 < spatialSize and a1 < spatialSize) {
            int n = a0 * spatialSize + a1;
            if (grid.mp.find(n) == grid.mp.end()) {
              grid.mp[n] = nSpatialSites++;
              for (int k = 0; k < nInputFeatures; k++)
                features.push_back(sW[J][k]);
            }
          }
        }
      }
    }
    break;
  case SpaceTime3d: {
    std::vector<arma::mat> csp;
    int cspLengths = 0;
    for (unsigned int i = 0; i < ops.size(); i++) {
      csp.push_back(constantSpeed(ops[i], 0.1));
      cspLengths += csp[i].n_rows;
    }
    // if (cspLengths>maxLength) {
    //   std::cout <<cspLengths*penSpeed3d << " " << std::flush;
    //   maxLength=cspLengths;
    // }
    int z = 0;
    for (unsigned int i = 0; i < ops.size(); i++) {
      for (unsigned int j = 0; j < csp[i].n_rows; ++j) {
        int a0 = csp[i](j, 0) + spatialSize / 2.0,
            a1 = csp[i](j, 1) + spatialSize / 2.0,
            a2 = renderSize * ((z - cspLengths / 2) * penSpeed3d + offset3d) +
                 spatialSize / 2;
        z++;
        if (a0 >= 0 and a1 >= 0 and a2 >= 0 and a0 < spatialSize and
            a1 < spatialSize and a2 < spatialSize) {
          int n = a0 * spatialSize * spatialSize + a1 * spatialSize + a2;
          if (grid.mp.find(n) == grid.mp.end()) {
            grid.mp[n] = nSpatialSites++;
            features.resize(nInputFeatures * nSpatialSites, 0);
          }
          ++features[grid.mp[n]];
        }
      }
    }
  } break;
  case VectorSpaceTime3d: {
    std::vector<arma::mat> csp;
    int cspLengths = 0;
    for (unsigned int i = 0; i < ops.size(); i++) {
      csp.push_back(constantSpeed(ops[i], 0.1));
      cspLengths += csp[i].n_rows;
    }
    // if (cspLengths>maxLength) {
    //   std::cout << cspLengths*penSpeed3d << " " << std::flush;
    //   maxLength=cspLengths;
    // }
    int z = 0;
    for (unsigned int i = 0; i < ops.size(); i++) {
      for (unsigned int j = 0; j < csp[i].n_rows - 1; ++j) {
        int a0 = csp[i](j, 0) + spatialSize / 2.0,
            a1 = csp[i](j, 1) + spatialSize / 2.0,
            a2 = renderSize * ((z - cspLengths / 2) * penSpeed3d + offset3d) +
                 spatialSize / 2;
        z++;
        if (a0 >= 0 and a1 >= 0 and a2 >= 0 and a0 < spatialSize and
            a1 < spatialSize and a2 < spatialSize) {
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
        }
      }
    }
  } break;
  }
}

void OnlineHandwritingPicture::draw(int spatialSize) {
  std::vector<char> grid(spatialSize * spatialSize, ' ');
  for (unsigned int i = 0; i < ops.size(); i++) {
    arma::mat csp = constantSpeed(ops[i], 0.1);
    for (unsigned int j = 0; j < csp.n_rows; ++j) {
      int a0 = csp(j, 0) + spatialSize / 2.0,
          a1 = csp(j, 1) + spatialSize / 2.0;
      if (a0 >= 0 and a1 >= 0 and a0 < spatialSize and a1 < spatialSize) {
        int n = a0 * spatialSize + a1;
        grid[n] = 'x';
      }
    }
  }
  for (int x = 0; x < spatialSize; x++)
    std::cout << "--";
  std::cout << "|\n";
  for (int y = 0; y < spatialSize; y++) {
    for (int x = 0; x < spatialSize; x++)
      std::cout << grid[x * spatialSize + y] << grid[x * spatialSize + y];
    std::cout << "|\n";
  }
  for (int x = 0; x < spatialSize; x++)
    std::cout << "--";
  std::cout << "|\n";
}
