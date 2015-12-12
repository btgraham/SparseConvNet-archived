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
  for (int x = 0; x < mat.cols; x++) {
    for (int y = 0; y < mat.rows; y++) {
      int j = x * mat.channels() + y * mat.channels() * mat.cols;
      bool interestingPixel = false;
      for (int i = 0; i < mat.channels(); i++)
        if (std::abs(matData[i + j] - backgroundColor) > 2)
          interestingPixel = true;
      if (interestingPixel) {
        for (int i = 0; i < mat.channels(); i++)
          matData[i + j] +=
              delta1[i] + delta2[i] * (matData[i + j] - backgroundColor) +
              delta3[i] * (x + xOffset) + delta4[i] * (y + yOffset);
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
void OpenCVPicture::jiggleFit(RNG &rng, int subsetSize, float minFill) {
  if (minFill < 0) { // Just pick a random subsetSize x subsetSquare that
                     // overlaps the picture as much as possible.
    if (mat.cols >= subsetSize)
      xOffset = -rng.randint(mat.cols - subsetSize + 1) - subsetSize / 2;
    else
      xOffset = rng.randint(subsetSize - mat.cols + 1) - subsetSize / 2;
    if (mat.rows >= subsetSize)
      yOffset = -rng.randint(mat.rows - subsetSize + 1) - subsetSize / 2;
    else
      yOffset = rng.randint(subsetSize - mat.rows + 1) - subsetSize / 2;
  } else {
    int fitCtr = 100; // Give up after 100 failed attempts to find a good fit
    bool goodFit = false;
    while (!goodFit and fitCtr-- > 0) {
      xOffset = -subsetSize / 2 -
                rng.randint(mat.cols - subsetSize); //-rng.randint(mat.cols);
      yOffset = -subsetSize / 2 -
                rng.randint(mat.rows - subsetSize); //-rng.randint(mat.rows);
      int pointsCtr = 0;
      int interestingPointsCtr = 0;
      for (int X = 5; X < subsetSize; X += 10) {
        for (int Y = 5; Y < subsetSize; Y += 10) {
          int x = X - xOffset - subsetSize / 2;
          int y = Y - yOffset - subsetSize / 2;
          pointsCtr++;
          if (0 <= x and x < mat.cols and 0 <= y and y < mat.rows)
            interestingPointsCtr +=
                (mat.ptr()[(pointsCtr % mat.channels()) + x * mat.channels() +
                           y * mat.channels() * mat.cols] != backgroundColor);
        }
      }
      assert(pointsCtr >= 10);
      if (interestingPointsCtr > pointsCtr * minFill)
        goodFit = true;
    }
    if (!goodFit) {
      std::cout << filename << " " << std::flush;
      xOffset = -mat.cols / 2 - 16 + rng.randint(32);
      yOffset = -mat.rows / 2 - 16 + rng.randint(32);
    }
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
void OpenCVPicture::loadData(int scale, int flags) {
  readImage(filename, mat, flags);
  float s = scale * 1.0f / std::min(mat.rows, mat.cols);
  transformImage(mat, backgroundColor, s, 0, 0, s);
  xOffset = -mat.cols / 2;
  yOffset = -mat.rows / 2;
}
void OpenCVPicture::loadDataWithoutScaling(int flags) {
  readImage(filename, mat, flags);
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
  int x0 = std::max(0, -xOffset - spatialSize / 2); // If x0<=x<x1 and y0<=y<y1
                                                    // then the (x,y)-th pixel
                                                    // is in the CNN's visual
                                                    // field.
  int x1 = std::min(mat.cols, spatialSize - xOffset - spatialSize / 2);
  int y0 = std::max(0, -yOffset - spatialSize / 2);
  int y1 = std::min(mat.rows, spatialSize - yOffset - spatialSize / 2);
  float *matData = ((float *)(mat.data));
  for (int x = x0; x < x1; x++) {
    for (int y = y0; y < y1; y++) {
      int j = x * mat.channels() + y * mat.channels() * mat.cols;
      bool interestingPixel =
          false; // Check pixel differs from the background color
      for (int i = 0; i < mat.channels(); i++)
        if (std::abs(matData[i + j] - backgroundColor) > 2)
          interestingPixel = true;
      if (interestingPixel) {
        int n = (x + xOffset + spatialSize / 2) * spatialSize +
                (y + yOffset +
                 spatialSize / 2); // Determine location in the input field.
        grid.mp[n] = nSpatialSites++;
        for (int i = 0; i < mat.channels(); i++) {
          features.push_back(scaleUCharColor(matData[i + j]));
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
