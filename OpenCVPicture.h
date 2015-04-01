#include "readImageToMat.h"

//If filename contain a image filename, then it gets loaded and then freed as necessary, unless mat already has somthing in it.

class OpenCVPicture : public Picture {
public:
  int xOffset; //Shift to the right
  int yOffset; //Shift down
  int backgroundColor;
  int scale;
  float scale2;
  float scale2xx, scale2xy,scale2yy;
  cv::Mat mat;
  string filename;

  void codifyInputData (SpatiallySparseBatch &batch);
  Picture* distort (RNG& rng, batchType type=TRAINBATCH);

  OpenCVPicture(int xSize, int ySize, int nInputFeatures, unsigned char backgroundColor,int label_ = -1) :
    backgroundColor(backgroundColor) {
    label=label_;
    xOffset=-xSize/2;
    yOffset=-ySize/2;
    mat.create(xSize,ySize, CV_8UC(nInputFeatures));
  }
  OpenCVPicture(string filename, int scale=256, unsigned char backgroundColor=128, int label_ = -1) :
    filename(filename), scale(scale), backgroundColor(backgroundColor) {
    label=label_;
  }

  ~OpenCVPicture() {}
  void jiggle(RNG &rng, int offlineJiggle) {
    xOffset+=rng.randint(offlineJiggle*2+1)-offlineJiggle;
    yOffset+=rng.randint(offlineJiggle*2+1)-offlineJiggle;
  }
  void randomCrop(RNG &rng, int subsetSize) {
    assert(subsetSize<=min(mat.rows,mat.cols));
    cropImage(mat, rng.randint(mat.cols-subsetSize),rng.randint(mat.rows-subsetSize), subsetSize, subsetSize);
    xOffset=yOffset=-subsetSize/2;
  }
  void affineTransform(float c00, float c01, float c10, float c11) {

    transformImage(mat, backgroundColor, c00, c01, c10, c11);
    xOffset=-mat.cols/2;
    yOffset=-mat.rows/2;
  }
  void jiggleFit2(RNG &rng, int subsetSize) {
    if (mat.cols>=subsetSize)
      xOffset=-rng.randint(mat.cols-subsetSize+1)-subsetSize/2;
    else
      xOffset=rng.randint(subsetSize-mat.cols+1)-subsetSize/2;
    if (mat.cols>=subsetSize)
      yOffset=-rng.randint(mat.rows-subsetSize+1)-subsetSize/2;
    else
      yOffset=rng.randint(subsetSize-mat.rows+1)-subsetSize/2;
  }
  void centerMass() {
    float ax=0, ay=0, axx=0, ayy=0, axy=0, d=0.001;
    for (int i=0; i<mat.channels(); i++) {
      for (int x=0;x<mat.cols;++x) {
        for (int y=0;y<mat.rows;++y) {
          float f=powf(backgroundColor-mat.ptr()[i+x*mat.channels()+y*mat.channels()*mat.cols],2);
          ax+=x*f;
          ay+=y*f;
          axx+=x*x*f;
          axy+=x*y*f;
          ayy+=y*y*f;
          d+=f;
        }
      }
    }
    ax/=d;
    ay/=d;
    axx/=d;
    axy/=d;
    ayy/=d;
    xOffset=-ax/2;
    yOffset=-ay/2;
    scale2xx=axx-ax*ax;
    scale2xy=axy-ax*ay;
    scale2yy=ayy-ay*ay;
    scale2=powf(scale2xx+scale2yy,0.5);
  }
  void loadDataOnceIgnoreScale(int flag=-1) {
    readImage(filename,mat,flag);
    xOffset=-mat.cols/2;
    yOffset=-mat.rows/2;
  }
  void loadData
  (int scale_=-1) {
    if (scale_==-1) scale_=scale;
    if (mat.empty()) {
      readTransformedImage(filename,mat,scale_,1,0,0,1,backgroundColor);
      xOffset=-mat.cols/2;
      yOffset=-mat.rows/2;
    }
  }
  float scaleUCharColor(float col) {
    float div=max(255-backgroundColor,backgroundColor);
    return (col-backgroundColor)/div;
  }
  string identify() {
    return filename;
  }
};

void OpenCVPicture::codifyInputData(SpatiallySparseBatch &batch) {
  loadData();
  for  (int i=0; i<mat.channels(); i++)
    batch.interfaces[0].features.hVector().push_back(0); //Background feature
  int backgroundNullVectorNumber=batch.interfaces[0].nSpatialSites++;
  batch.interfaces[0].backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
  vector<int> grid(batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize,backgroundNullVectorNumber);
  for (int x=0; x<mat.cols; x++) {
    int X=x+xOffset+batch.interfaces[0].spatialSize/2;
    for (int y=0; y<mat.rows; y++) {
      int Y=y+yOffset+batch.interfaces[0].spatialSize/2;
      if (X>=0 && X<batch.interfaces[0].spatialSize && Y>=0 && Y<batch.interfaces[0].spatialSize) {
        bool flag=false;
        for (int i=0; i<mat.channels(); i++)
          if (abs(scaleUCharColor(mat.ptr()[i+x*mat.channels()+y*mat.channels()*mat.cols]))>0.02)
            flag=true;
        if (flag) {
          int n=X*batch.interfaces[0].spatialSize+Y;
          grid[n]=batch.interfaces[0].nSpatialSites++;
          for (int i=0; i<mat.channels(); i++) {
            batch.interfaces[0].features.hVector().push_back
              (scaleUCharColor(mat.ptr()[i+x*mat.channels()+y*mat.channels()*mat.cols]));
          }
        }
      }
    }
  }
  batch.interfaces[0].grids.push_back(grid);
  batch.interfaces[0].batchSize++;
  batch.labels.hVector().push_back(label);
}

// int xSize;  mat.cols
// int ySize;  mat.rows
// int nInputFeatures; mat.channels()




//Example distortion functions

// // Picture* OpenCVPicture::distort(RNG& rng) {
// //   OpenCVPicture* pic=new OpenCVPicture(*this);
// //   pic->loadData();
// //   pic->jiggle(rng,100);
// //   return pic;
// // }


// // Picture* OpenCVPicture::distort(RNG& rng) {
// //   OpenCVPicture* pic=new OpenCVPicture(filename,label);
// //   float c00=1;
// //   float c01=0;
// //   float c10=0;
// //   float c11=1;
// //   c00*=1+rng.uniform(-0.2,0.2); // x stretch
// //   c11*=1+rng.uniform(-0.2,0.2); // y stretch
// //   if (rng.randint(2)==0) c00*=-1; //Horizontal flip
// //   int r=rng.randint(3);
// //   float alpha=rng.uniform(-0.2,0.2);
// //   if (r==0) {c01+=alpha*c00;c11+=alpha*c10;}
// //   if (r==1) {c10+=alpha*c00;c11+=alpha*c01;}
// //   if (r==2) {
// //     float c=cos(alpha); float s=sin(alpha);
// //     float t00=c00*c-c01*s; float t01=c00*s+c01*c; c00=t00;c01=t01;
// //     float t10=c10*c-c11*s; float t11=c10*s+c11*c; c10=t10;c11=t11;}
// //   pic->loadData(c00,c01,c10,c11);
// //   pic->jiggle(rng,64);
// //   return pic;
// // }



// // Picture* OpenCVPicture::distort(RNG& rng) {
// //   OpenCVPicture* pic=new OpenCVPicture(filename,label);
// //   float c00=1;
// //   float c01=0;
// //   float c10=0;
// //   float c11=1;
// //   if (rng.randint(2)==0) c00*=-1; //Horizontal flip
// //   float alpha=rng.uniform(-0.2,0.2);
// //   {float c=cos(alpha); float s=sin(alpha);
// //     float t00=c00*c-c01*s; float t01=c00*s+c01*c; c00=t00;c01=t01;
// //     float t10=c10*c-c11*s; float t11=c10*s+c11*c; c10=t10;c11=t11;}
// //   pic->loadData(c00,c01,c10,c11);

// //   OpenCVPicture* pic2=new OpenCVPicture(pic->mat.cols+0,pic->mat.rows+0,label);
// //   EDfield edf(20,600,3,17);
// //   for (int y=0; y<pic2->mat.rows; y++)
// //     for (int x=0; x<pic2->mat.cols;x++) {
// //       FloatPoint p(x+pic2->xOffset+0.5,y+pic2->yOffset+0.5);
// //       p.stretch(edf);
// //       p.stretch(edf);
// //       p.stretch(edf);
// //       for (int i=0; i<nInputFeatures; i++)
// //         pic2->bitmap[x+y*pic2->mat.cols+i*pic2->mat.cols*pic2->mat.rows]=pic->interpolate(p, i);
// //     }
// //   pic->unloadData();
// //   delete pic;
// //   pic2->jiggle(rng,150);
// //   return pic2;
// // }
