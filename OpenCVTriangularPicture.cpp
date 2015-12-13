#include "OpenCVPicture.h"
#include <cmath>

float OpenCVPicture::scaleUCharColor(float col) {
  float div = std::max(255 - backgroundColor, backgroundColor);
  return (col - backgroundColor) / div;
}

OpenCVPicture::OpenCVPicture(int xSize, int ySize, int nInputFeatures,
                             unsigned char backgroundColor, int label)
    : Picture(label), backgroundColor(backgroundColor) {
  xOffset = -xSize / 2;
  yOffset = -ySize / 2;
  mat.create(xSize, ySize, CV_32FC(nInputFeatures));
}
OpenCVPicture::OpenCVPicture(std::string filename,
                             unsigned char backgroundColor, int label_)
    : filename(filename), backgroundColor(backgroundColor) {
  label = label_;
}

OpenCVPicture::~OpenCVPicture() {}

void OpenCVPicture::jiggle(RNG &rng, int offlineJiggle) {
  xOffset += rng.randint(offlineJiggle * 2 + 1) - offlineJiggle;
  yOffset += rng.randint(offlineJiggle * 2 + 1) - offlineJiggle;
}
void OpenCVPicture::colorDistortion(RNG &rng, int sigma1, int sigma2,
                                    int sigma3, int sigma4) {
  // Call as a final preprocessing step, after any affine transforms and
  // jiggling.
  assert(mat.type() % 8 == 5); // float
  std::vector<float> delta1(mat.channels());
  std::vector<float> delta2(mat.channels());
  std::vector<float> delta3(mat.channels());
  std::vector<float> delta4(mat.channels());
  for (int j = 0; j < mat.channels(); j++) {
    delta1[j] = rng.normal(0, sigma1);
    delta2[j] = rng.normal(0, sigma2);
    delta3[j] = rng.normal(0, sigma3);
    delta4[j] = rng.normal(0, sigma4);
  }
  float *matData = ((float *)(mat.data));
  for (int y = 0; y < mat.rows; y++) {
    for (int x = 0; x < mat.cols; x++) {
      int j = x * mat.channels() + y * mat.channels() * mat.cols;
      bool interestingPixel = false;
      for (int i = 0; i < mat.channels(); i++)
        if (std::abs(matData[i + j] - backgroundColor) > 2)
          interestingPixel = true;
      if (interestingPixel) {
        for (int i = 0; i < mat.channels(); i++)
          matData[i + j] +=
              delta1[i] + delta2[i] * (matData[i + j] - backgroundColor) +
              delta3[i] * (x - mat.cols / 2) + delta4[i] * (y - mat.rows / 2);
      }
    }
  }
}
void OpenCVPicture::randomCrop(RNG &rng, int subsetSize) {
  assert(subsetSize <= std::min(mat.rows, mat.cols));
  cropImage(mat, rng.randint(mat.cols - subsetSize),
            rng.randint(mat.rows - subsetSize), subsetSize, subsetSize);
  xOffset = yOffset = -subsetSize / 2;
}
void OpenCVPicture::affineTransform(float c00, float c01, float c10,
                                    float c11) {

  transformImage(mat, backgroundColor, c00, c01, c10, c11);
  xOffset = -mat.cols / 2;
  yOffset = -mat.rows / 2;
}
void OpenCVPicture::jiggleFit(
    RNG &rng, int subsetSize,
    float minFill) { // subsetSize==spatialSize for codifyInputData
  assert(minFill > 0);
  int fitCtr = 100; // Give up after 100 failed attempts to find a good fit
  bool goodFit = false;
  float *matData = ((float *)(mat.data));
  while (!goodFit and fitCtr-- > 0) {
    xOffset = -rng.randint(mat.cols - subsetSize / 3);
    yOffset = -rng.randint(mat.rows - subsetSize / 3);
    int pointsCtr = 0;
    int interestingPointsCtr = 0;
    for (int X = 5; X < subsetSize; X += 10) {
      for (int Y = 5; Y < subsetSize - X; Y += 10) {
        int x = X - xOffset - subsetSize / 3;
        int y = Y - yOffset - subsetSize / 3;
        pointsCtr++;
        if (0 <= x and x < mat.cols and 0 <= y and y < mat.rows) {
          interestingPointsCtr +=
              (matData[(pointsCtr % mat.channels()) + x * mat.channels() +
                       y * mat.channels() * mat.cols] != backgroundColor);
        }
      }
    }
    if (interestingPointsCtr > pointsCtr * minFill)
      goodFit = true;
  }
  if (!goodFit) {
    std::cout << filename << " " << std::flush;
    xOffset = -mat.cols / 2 - 16 + rng.randint(33);
    yOffset = -mat.rows / 2 - 16 + rng.randint(33);
  }
}
void OpenCVPicture::centerMass() {
  float ax = 0, ay = 0, axx = 0, ayy = 0, axy, d = 0.001;
  for (int i = 0; i < mat.channels(); i++) {
    for (int x = 0; x < mat.cols; ++x) {
      for (int y = 0; y < mat.rows; ++y) {
        float f = powf(backgroundColor -
                           mat.ptr()[i + x * mat.channels() +
                                     y * mat.channels() * mat.cols],
                       2);
        ax += x * f;
        ay += y * f;
        axx += x * x * f;
        axy += x * y * f;
        ayy += y * y * f;
        d += f;
      }
    }
  }
  ax /= d;
  ay /= d;
  axx /= d;
  axy /= d;
  ayy /= d;
  xOffset = -ax / 2;
  yOffset = -ay / 2;
  scale2xx = axx - ax * ax;
  scale2xy = axy - ax * ay;
  scale2yy = ayy - ay * ay;
  scale2 = powf(scale2xx + scale2yy, 0.5);
}
void OpenCVPicture::loadDataWithoutScaling(int flag) {
  readImage(filename, mat, flag);
  xOffset = -mat.cols / 2;
  yOffset = -mat.rows / 2;
}
void OpenCVPicture::loadData(int scale, int flags) {
  readImage(filename, mat, flags);
  float s = scale * 1.0f / std::min(mat.rows, mat.cols);
  transformImage(mat, backgroundColor, s, 0, 0, s);
  xOffset = -mat.cols / 2;
  yOffset = -mat.rows / 2;
}
void OpenCVPicture::loadDataWithoutScalingRemoveModalColor(int flags) {
  cv::Mat temp = cv::imread(filename, flags);
  if (temp.empty()) {
    std::cout << "Error : Image " << filename << " cannot be loaded..."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::vector<int> modalColor;
  for (int i = 0; i < temp.channels(); ++i) {
    int whereMax = 0;
    int m = 0;
    std::vector<int> counts(256, 0);
    for (int y = 0; y < temp.rows; y++) {
      for (int x = 0; x < temp.cols; x++) {
        int c =
            temp.ptr()[i + x * temp.channels() + y * mat.channels() * mat.cols];
        counts[c]++;
        if (counts[c] > m) {
          m = counts[c];
          whereMax = c;
        }
      }
    }
    modalColor.push_back(whereMax);
  }
  temp.convertTo(mat, CV_32FC(temp.channels()));
  float *matData = ((float *)(mat.data));
  for (int i = 0; i < mat.channels(); ++i)
    for (int y = 0; y < temp.rows; y++)
      for (int x = 0; x < temp.cols; x++)
        matData[i + x * mat.channels() + y * mat.channels() * mat.cols] -=
            modalColor[i];
  backgroundColor = 0;
  xOffset = -mat.cols / 2;
  yOffset = -mat.rows / 2;
}

std::string OpenCVPicture::identify() { return filename; }

void OpenCVPicture::codifyInputData(SparseGrid &grid,
                                    std::vector<float> &features,
                                    int &nSpatialSites, int spatialSize) {
  assert(!mat.empty());
  assert(mat.type() % 8 == 5);
  for (int i = 0; i < mat.channels(); i++)
    features.push_back(0); // Background feature
  grid.backgroundCol = nSpatialSites++;
  float *matData = ((float *)(mat.data));
  for (int x = 0; x < mat.cols; x++) {
    int X = x + xOffset + spatialSize / 3;
    for (int y = 0; y < mat.rows; y++) {
      int Y = y + yOffset + spatialSize / 3;
      if (X >= 0 && Y >= 0 && X + Y < spatialSize) {
        bool flag = false;
        for (int i = 0; i < mat.channels(); i++)
          if (std::abs(scaleUCharColor(
                  matData[i + x * mat.channels() +
                          y * mat.channels() * mat.cols])) > 0.02)
            flag = true;
        if (flag) {
          int n = X * spatialSize + Y;
          grid.mp[n] = nSpatialSites++;
          for (int i = 0; i < mat.channels(); i++) {
            features.push_back(
                scaleUCharColor(matData[i + x * mat.channels() +
                                        y * mat.channels() * mat.cols]));
          }
        }
      }
    }
  }
}

void matrixMul2x2inPlace(float &c00, float &c01, float &c10, float &c11,
                         float a00, float a01, float a10, float a11) { // c<-c*a
  float t00 = c00 * a00 + c01 * a10;
  float t01 = c00 * a01 + c01 * a11;
  float t10 = c10 * a00 + c11 * a10;
  float t11 = c10 * a01 + c11 * a11;
  c00 = t00;
  c01 = t01;
  c10 = t10;
  c11 = t11;
}
