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
                             unsigned char backgroundColor, int label)
    : Picture(label), filename(filename), backgroundColor(backgroundColor) {}

OpenCVPicture::~OpenCVPicture() {}

void OpenCVPicture::jiggle(RNG &rng, int offlineJiggle) {
  xOffset += rng.randint(offlineJiggle * 2 + 1) - offlineJiggle;
  yOffset += rng.randint(offlineJiggle * 2 + 1) - offlineJiggle;
}
void OpenCVPicture::elasticDistortion(RNG &rng, float amplitude, float radius) {
  // http://research.microsoft.com/pubs/68920/icdar03.pdf
  // Best Practices for Convolutional Neural Networks Applied to Visual Document
  // Analysis; Patrice Y. Simard, Dave Steinkraus, John C. Platt

  // faster version??
  // cv::Mat t0 = cv::Mat::zeros(cv::Size(mat.cols / 10, mat.rows / 10),
  // CV_32FC1),
  //         map_x =
  //             cv::Mat::zeros(cv::Size(mat.cols / 10, mat.rows / 10),
  //             CV_32FC1),
  //         map_y =
  //             cv::Mat::zeros(cv::Size(mat.cols / 10, mat.rows / 10),
  //             CV_32FC1);
  // cv::theRNG().state = rng.gen();
  // cv::randn(t0, 0, amplitude * radius);
  // cv::GaussianBlur(t0, map_x, cv::Size(0, 0), radius / 10);
  // cv::randn(t0, 0, amplitude * radius);
  // cv::GaussianBlur(t0, map_y, cv::Size(0, 0), radius / 10);
  // for (int j = 0; j < map_x.rows; j++) {
  //   for (int i = 0; i < map_x.cols; i++) {
  //     map_x.at<float>(j, i) += 4.5 + i * mat.cols * 1.0 / map_x.cols;
  //     map_y.at<float>(j, i) += 4.5 + j * mat.rows * 1.0 / map_x.rows;
  //   }
  // }
  // cv::Mat map_X, map_Y;
  // cv::resize(map_x, map_X, mat.size());
  // cv::resize(map_y, map_Y, mat.size());
  // {
  //   cv::Mat temp;
  //   cv::remap(mat, temp, map_X, map_Y, CV_INTER_LINEAR, IPL_BORDER_CONSTANT,
  //             cv::Scalar(backgroundColor, backgroundColor, backgroundColor));
  //   mat = temp;
  // }

  // naive method(not much slower?)
  cv::Mat t0 = cv::Mat::zeros(cv::Size(mat.cols / 1, mat.rows / 1), CV_32FC1),
          map_x =
              cv::Mat::zeros(cv::Size(mat.cols / 1, mat.rows / 1), CV_32FC1),
          map_y =
              cv::Mat::zeros(cv::Size(mat.cols / 1, mat.rows / 1), CV_32FC1);
  cv::theRNG().state = rng.gen();
  cv::randn(t0, 0, amplitude * radius);
  cv::GaussianBlur(t0, map_x, cv::Size(0, 0), radius);
  cv::randn(t0, 0, amplitude * radius);
  cv::GaussianBlur(t0, map_y, cv::Size(0, 0), radius);

  for (int j = 0; j < map_x.rows; j++) {
    for (int i = 0; i < map_x.cols; i++) {
      map_x.at<float>(j, i) += i;
      map_y.at<float>(j, i) += j;
    }
  }
  {
    cv::Mat temp;
    cv::remap(mat, temp, map_x, map_y, CV_INTER_LINEAR, IPL_BORDER_CONSTANT,
              cv::Scalar(128, 128, 128));
    mat = temp;
  }
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
void OpenCVPicture::jiggleFit(RNG &rng, int subsetSize, float minFill) {
  for (int fitCtr = 100; // Give up after 100 failed attempts to find a good fit
       fitCtr > 0; fitCtr--) {
    {
      if (mat.cols >= subsetSize)
        xOffset = -rng.randint(mat.cols - subsetSize + 1) - subsetSize / 2;
      else
        xOffset = rng.randint(subsetSize - mat.cols + 1) - subsetSize / 2;
      if (mat.rows >= subsetSize)
        yOffset = -rng.randint(mat.rows - subsetSize + 1) - subsetSize / 2;
      else
        yOffset = rng.randint(subsetSize - mat.rows + 1) - subsetSize / 2;
    }
    if (minFill < 0) {
      fitCtr = -1; // Just take any old crop
    } else {
      int pointsCtr = 0;
      int interestingPointsCtr = 0;
      // If x0<=x<x1 and y0<=y<y1 then the (x,y)-th pixel is in the CNN's visual
      // field.
      int x0 = std::max(0, -xOffset - subsetSize / 2);
      int x1 = std::min(mat.cols, subsetSize - xOffset - subsetSize / 2);
      int y0 = std::max(0, -yOffset - subsetSize / 2);
      int y1 = std::min(mat.rows, subsetSize - yOffset - subsetSize / 2);
      float *matData = ((float *)(mat.data));
      assert(subsetSize > 20);
      int subsample = subsetSize / 20;
      for (int x = x0 + subsample / 2; x < x1; x += subsample) {
        for (int y = y0 + subsample / 2; y < y1; y += subsample) {
          pointsCtr++;
          int j = x * mat.channels() + y * mat.channels() * mat.cols;
          for (int i = 0; i < mat.channels(); i++)
            if (std::abs(matData[i + j] - backgroundColor) > 2) {
              interestingPointsCtr++;
              break;
            }
        }
      }
      if (interestingPointsCtr > pointsCtr * minFill)
        fitCtr = -1;
      if (fitCtr == 0) {
        std::cout << filename << " " << std::flush;
        xOffset = -mat.cols / 2 - 16 + rng.randint(32);
        yOffset = -mat.rows / 2 - 16 + rng.randint(32);
      }
    }
  }
}
void OpenCVPicture::centerMass() {
  float ax = 0, ay = 0, axx = 0, ayy = 0, axy = 0, d = 0.001;
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
void OpenCVPicture::blur(float radius) {
  cv::Mat temp = mat;
  cv::GaussianBlur(temp, mat, cv::Size(0, 0), radius);
}
void OpenCVPicture::addSpatiallyCoherentNoise(RNG &rng, float amplitude,
                                              float radius) {
  cv::Mat t0 = mat.clone(), t1 = mat.clone();
  cv::theRNG().state = rng.gen();
  cv::randn(t0, 0, 1);
  cv::GaussianBlur(t0, t1, cv::Size(0, 0), radius);
  mat += (amplitude * radius) * t1;
  //  t0 = mat;
  // cv::addWeighted(t0, 1, t1, amplitude / radius, 0, mat);
}
void OpenCVPicture::multiplySpatiallyCoherentNoise(RNG &rng, float amplitude,
                                                   float radius) {
  // cv::Mat s0 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC3),
  //         t0 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC3),
  //         s1 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC3),
  //         t1 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC3);
  // cv::theRNG().state = rng.gen();
  // cv::randn(s0, 0, amplitude * radius);
  // cv::randn(s1, 0, amplitude * radius);
  // cv::GaussianBlur(s0, t0, cv::Size(0, 0), radius);
  // cv::GaussianBlur(s1, t1, cv::Size(0, 0), radius);
  // s0 = cv::max(t0, 0);
  // s1 = cv::max(t1, 0);
  // mat = mat - mat.mul(s0) + (cv::Scalar(255, 255, 255) - mat).mul(s1);

  cv::Mat s0 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC1),
          t0 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC1),
          s1 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC1),
          t1 = cv::Mat::zeros(cv::Size(mat.cols, mat.rows), CV_32FC1);
  cv::theRNG().state = rng.gen();
  cv::randn(s0, 0, amplitude * radius);
  cv::randn(s1, 0, amplitude * radius);
  cv::GaussianBlur(s0, t0, cv::Size(0, 0), radius);
  cv::GaussianBlur(s1, t1, cv::Size(0, 0), radius);
  s0 = cv::max(t0, 0);
  s1 = cv::max(t1, 0);
  cv::cvtColor(s0, t0, CV_GRAY2RGB);
  cv::cvtColor(s1, t1, CV_GRAY2RGB);
  mat = mat - mat.mul(t0) + (cv::Scalar(255, 255, 255) - mat).mul(t1);
}
void OpenCVPicture::loadData(int scale, int flags) {
  loadDataWithoutScaling(flags);
  float s = scale * 1.0f / std::min(mat.rows, mat.cols);
  transformImage(mat, backgroundColor, s, 0, 0, s);
  xOffset = -mat.cols / 2;
  yOffset = -mat.rows / 2;
}
void OpenCVPicture::loadDataWithoutScaling(int flags) {
  if (!rawData.empty()) {
    cv::Mat temp = cv::imdecode(rawData, flags);
    temp.convertTo(mat, CV_32FC(temp.channels()));
  } else {
    cv::Mat temp = cv::imread(filename, flags);
    if (temp.empty()) {
      std::cout << "Error : Image " << filename << " cannot be loaded..."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    temp.convertTo(mat, CV_32FC(temp.channels()));
  }
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
        int c = temp.ptr()[i + x * temp.channels() +
                           y * temp.channels() * temp.cols];
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
void OpenCVPicture::loadDataWithoutScalingRemoveMeanColor(int flags) {
  cv::Mat temp = cv::imread(filename, flags);
  if (temp.empty()) {
    std::cout << "Error : Image " << filename << " cannot be loaded..."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::vector<float> meanColor(temp.channels());
  for (int i = 0; i < temp.channels(); ++i) {
    for (int y = 0; y < temp.rows; y++) {
      for (int x = 0; x < temp.cols; x++) {
        int c = temp.ptr()[i + x * temp.channels() +
                           y * temp.channels() * temp.cols];
        meanColor[i] += c;
      }
    }
  }
  for (int i = 0; i < temp.channels(); ++i)
    meanColor[i] /= temp.rows * temp.cols;
  temp.convertTo(mat, CV_32FC(temp.channels()));
  float *matData = ((float *)(mat.data));
  for (int i = 0; i < mat.channels(); ++i)
    for (int y = 0; y < temp.rows; y++)
      for (int x = 0; x < temp.cols; x++)
        // matData[i + x * mat.channels() + y * mat.channels() * mat.cols] -=
        //     meanColor[i];
        matData[i + x * mat.channels() + y * mat.channels() * mat.cols] =
            128 +
            (matData[i + x * mat.channels() + y * mat.channels() * mat.cols] -
             meanColor[i]) /
                2;
  backgroundColor = 128;
  xOffset = -mat.cols / 2;
  yOffset = -mat.rows / 2;
}

int OpenCVPicture::area() {
  assert(mat.type() % 8 == 5); // float
  int area = 0;
  float *matData = ((float *)(mat.data));
  for (int y = 0; y < mat.rows; y++) {
    for (int x = 0; x < mat.cols; x++) {
      int j = x * mat.channels() + y * mat.channels() * mat.cols;
      bool interestingPixel = false;
      for (int i = 0; i < mat.channels(); i++)
        if (std::abs(matData[i + j] - backgroundColor) > 2)
          interestingPixel = true;
      if (interestingPixel)
        ++area;
    }
  }
  return area;
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
  // If x0<=x<x1 and y0<=y<y1 then the (x,y)-th pixel is in the CNN's visual
  // field.
  int x0 = std::max(0, -xOffset - spatialSize / 2);
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
