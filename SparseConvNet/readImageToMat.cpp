#include "readImageToMat.h"

void readImage(std::string filename, cv::Mat &mat, int flags) {
  cv::Mat temp = cv::imread(filename, flags);
  if (temp.empty()) {
    std::cout << "Error : Image " << filename << " cannot be loaded..."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  temp.convertTo(mat, CV_32FC(temp.channels()));
}

void writeImage(cv::Mat &mat, int n) {
  cv::Mat temp;
  mat.convertTo(temp, CV_8UC(mat.channels()));
  std::string filename = std::string("OpenCVwriteImage-") + std::to_string(n) +
                         std::string(".png");
  std::cout << "Writing " << filename << "\n";
  cv::imwrite(filename.c_str(), temp);
}
void writeImage(cv::Mat &mat, std::string filename) {
  cv::Mat temp;
  mat.convertTo(temp, CV_8UC(mat.channels()));
  std::cout << "Writing " << filename << "\n";
  cv::imwrite(filename.c_str(), temp);
}

void transformImage(cv::Mat &src, int backgroundColor, float c00, float c01,
                    float c10, float c11) {

  cv::Mat warp, dst;
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];

  int X = src.cols;
  int Y = src.rows;
  int x = 0;
  int y = 0;
  srcTri[0] = cv::Point2f(x, y);
  srcTri[1] = cv::Point2f(X, y);
  srcTri[2] = cv::Point2f(x, Y);
  dstTri[0] = cv::Point2f(x * c00 + y * c10, x * c01 + y * c11);
  dstTri[1] = cv::Point2f(X * c00 + y * c10, X * c01 + y * c11);
  dstTri[2] = cv::Point2f(x * c00 + Y * c10, x * c01 + Y * c11);
  float m;
  m = std::min(std::min(std::min(dstTri[0].x, dstTri[1].x), dstTri[2].x),
               dstTri[1].x + dstTri[2].x);
  dstTri[0].x -= m;
  dstTri[1].x -= m;
  dstTri[2].x -= m;
  m = std::min(std::min(std::min(dstTri[0].y, dstTri[1].y), dstTri[2].y),
               dstTri[1].y + dstTri[2].y);
  dstTri[0].y -= m;
  dstTri[1].y -= m;
  dstTri[2].y -= m;
  dst = cv::Mat::zeros(
      ceil(std::max(std::max(std::max(dstTri[0].y, dstTri[1].y), dstTri[2].y),
                    dstTri[1].y + dstTri[2].y)),
      ceil(std::max(std::max(std::max(dstTri[0].x, dstTri[1].x), dstTri[2].x),
                    dstTri[1].x + dstTri[2].x)),
      src.type());
  warp = cv::getAffineTransform(srcTri, dstTri);
  cv::warpAffine(src, dst, warp, dst.size(), cv::INTER_AREA,
                 cv::BORDER_CONSTANT,
                 cv::Scalar(backgroundColor, backgroundColor, backgroundColor));
  src = dst;
}

void cropImage(cv::Mat &src, int X, int Y, int Width, int Height) {
  src = src(cv::Rect(X, Y, Width, Height)).clone();
}
