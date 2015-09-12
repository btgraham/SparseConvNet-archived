#include "readImageToMat.h"

void readImage(std::string filename, cv::Mat& mat, int flags) {
  cv::Mat temp = cv::imread(filename, flags);
  if (temp.empty()) {
    std::cout << "Error : Image " << filename << " cannot be loaded..." << std::endl;
    exit(EXIT_FAILURE);
  }
  temp.convertTo(mat,CV_32FC(temp.channels()));
}

void writeImage(cv::Mat& mat,int n) {
  cv::Mat temp;
  mat.convertTo(temp,CV_8UC(mat.channels()));
  std::string filename=std::string("OpenCVwriteImage-")+std::to_string(n)+std::string(".png");
  cv::imwrite(filename.c_str(),temp);
}
void writeImage(cv::Mat& mat,std::string filename) {
  cv::Mat temp;
  mat.convertTo(temp,CV_8UC(mat.channels()));
  cv::imwrite(filename.c_str(),temp);
}

void readTransformedImage(std::string filename, cv::Mat& dst, float scale, int flags,
                          float c00,float c01,float c10,float c11,
                          int backgroundColor,
                          int x, int X, int y, int Y) {
  cv::Mat src,warp;
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];

  readImage(filename,src,flags);
  if (X<0 || X>src.cols-1) X=src.cols;
  if (Y<0 || Y>src.cols-1) Y=src.rows;
  if (x<0) x=0;
  if (y<0) y=0;
  scale=scale/std::min(src.rows,src.cols);
  srcTri[0]=cv::Point2f(x,y);
  srcTri[1]=cv::Point2f(X,y);
  srcTri[2]=cv::Point2f(x,Y);
  dstTri[0]=cv::Point2f(x*c00*scale+y*c10*scale,x*c01*scale+y*c11*scale);
  dstTri[1]=cv::Point2f(X*c00*scale+y*c10*scale,X*c01*scale+y*c11*scale);
  dstTri[2]=cv::Point2f(x*c00*scale+Y*c10*scale,x*c01*scale+Y*c11*scale);
  float m;
  m=std::min(std::min(std::min(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x);
  dstTri[0].x-=m;
  dstTri[1].x-=m;
  dstTri[2].x-=m;
  m=std::min(std::min(std::min(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y);
  dstTri[0].y-=m;
  dstTri[1].y-=m;
  dstTri[2].y-=m;
  dst = cv::Mat::zeros(ceil(std::max(std::max(std::max(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y)),
                       ceil(std::max(std::max(std::max(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x)),
                       src.type());
  warp = cv::getAffineTransform( srcTri, dstTri );
  cv::warpAffine( src, dst, warp, dst.size(),cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(backgroundColor,backgroundColor,backgroundColor));
}

void transformImage(cv::Mat& src, int backgroundColor,float c00,float c01,float c10,float c11) {

  cv::Mat warp, dst;
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];

  int X=src.cols;
  int Y=src.rows;
  int x=0;
  int y=0;
  srcTri[0]=cv::Point2f(x,y);
  srcTri[1]=cv::Point2f(X,y);
  srcTri[2]=cv::Point2f(x,Y);
  dstTri[0]=cv::Point2f(x*c00+y*c10,x*c01+y*c11);
  dstTri[1]=cv::Point2f(X*c00+y*c10,X*c01+y*c11);
  dstTri[2]=cv::Point2f(x*c00+Y*c10,x*c01+Y*c11);
  float m;
  m=std::min(std::min(std::min(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x);
  dstTri[0].x-=m;
  dstTri[1].x-=m;
  dstTri[2].x-=m;
  m=std::min(std::min(std::min(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y);
  dstTri[0].y-=m;
  dstTri[1].y-=m;
  dstTri[2].y-=m;
  dst = cv::Mat::zeros(ceil(std::max(std::max(std::max(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y)),
                       ceil(std::max(std::max(std::max(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x)),
                       src.type());
  warp = cv::getAffineTransform( srcTri, dstTri );
  cv::warpAffine( src, dst, warp, dst.size(),cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(backgroundColor,backgroundColor,backgroundColor));
  src=dst;
}

void cropImage(cv::Mat& src, int X, int Y, int Width, int Height) {
  src= src(cv::Rect(X,Y,Width,Height)).clone();
}

void distortImageColor(cv::Mat& mat, RNG& rng, float sigma1, float sigma2, float sigma3, float sigma4) {
  assert(mat.type()%8==5); //float
  std::vector<float> delta1(mat.channels());
  std::vector<float> delta2(mat.channels());
  std::vector<float> delta3(mat.channels());
  std::vector<float> delta4(mat.channels());
  for (int j=0;j<mat.channels();j++) {
    delta1[j]=rng.normal(0,sigma1);
    delta2[j]=rng.normal(0,sigma2);
    delta3[j]=rng.normal(0,sigma3);
    delta4[j]=rng.normal(0,sigma4);
  }
  float* matData=((float*)(mat.data));
  int j=0;
  for (int y=0;y<mat.rows;++y) {
    for (int x=0;x<mat.cols;++x) {
      for (int i=0;i<mat.channels();++i) {
        matData[j]=matData[j]+
          delta1[i]+
          delta2[i]*(matData[j]/128-1)+
          delta3[i]*(x-mat.cols/2)+
          delta4[i]*(y-mat.rows/2);
        ++j;
      }
    }
  }
}
