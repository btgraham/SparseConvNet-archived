int nInputFeatures=9;
#include "OnlineHandwriting.h"

void OnlinePicture::codifyInputData(SpatiallySparseBatch &batch) {
  for(int i=0;i<nInputFeatures;i++)
    batch.interfaces[0].features.hVector().push_back(0);  //Background feature
  int backgroundNullVectorNumber=batch.interfaces[0].nSpatialSites++;
  batch.interfaces[0].backgroundNullVectorNumbers.push_back(backgroundNullVectorNumber);
  vector<int> grid(batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize,backgroundNullVectorNumber);
  float piBy4=3.1415926536/4;
  vector<float> w(9*batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize,0);
  for (int i=0; i<ops.size(); i++) {
    OnlinePenStroke csp=constantSpeed(ops[i],0.1);
    for (int j=0;j<csp.size()-1;j++) {
      int n=mapToGrid(csp[j].x,batch.interfaces[0].spatialSize)*batch.interfaces[0].spatialSize+mapToGrid(csp[j].y,batch.interfaces[0].spatialSize);
      float dx=(csp[j+1].x-csp[j].x)/0.5;
      float dy=(csp[j+1].y-csp[j].y)/0.5;
      float m=powf(dx*dx+dy*dy,0.5);
      dx/=m;
      dy/=m;
      w[9*n]=1;
      for (int k=0;k<8;k++) {
        float alpha=1-acosf(cosf(k*piBy4)*dx+sinf(k*piBy4)*dy)/piBy4;
        if (alpha>0)
          w[9*n+k+1]+=alpha;
      }
    }
  }
  for (int n=0;n<batch.interfaces[0].spatialSize*batch.interfaces[0].spatialSize;n++)
    if (w[9*n]==1) {
      grid[n]=batch.interfaces[0].nSpatialSites++;
      for (int k=0;k<9;k++)
        batch.interfaces[0].features.hVector().push_back(w[9*n+k]);
    }
  batch.interfaces[0].grids.push_back(grid);
  batch.interfaces[0].batchSize++;
  batch.labels.hVector().push_back(label);
}
