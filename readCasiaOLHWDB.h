//Obtain POT files from http://www.nlpr.ia.ac.cn/databases/handwriting/home.html and put them in a directory CASIA_pot_files/
const int nCharacters = 3755;

#include "gbcodes3755.h"
struct potCharacterHeader{
  unsigned short sampleSize;
  unsigned short label;
  unsigned short zzz;
  unsigned short nStrokes;
};
struct iPoint{
  short x;
  short y;
};


int readPotFile(vector<Picture*> &characters, const char* filename) {
  ifstream file(filename,ios::in|ios::binary);
  if (!file) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  potCharacterHeader pCH;
  iPoint iP;
  FloatPoint fP;
  while (file.read((char*)&pCH,sizeof(potCharacterHeader))) {
    OnlinePicture* character = new OnlinePicture();
    character->label=pCH.label;
    file.read((char*)&iP,sizeof(iPoint));
    while (iP.y!=-1) {
      OnlinePenStroke stroke;
      while (iP.x!=-1){
        fP.x=iP.x;
        fP.y=iP.y;
        stroke.push_back(fP);
        file.read((char*)&iP,sizeof(iPoint));}
      character->ops.push_back(stroke);
      file.read((char*)&iP,sizeof(iPoint));}
    normalize(character->ops);
    character->label=find(gbcodesPOT,gbcodesPOT+3755,character->label)-gbcodesPOT;
    if (character->label<nCharacters)
      characters.push_back(character);
    else
      delete character;
  }
  file.close();
  return 0;
}

SpatialDataset Casia11TrainSet() {
  SpatialDataset dataset;
  dataset.name="CASIA OLHWDB1.1 train set--240 writers";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=nInputFeatures;
  dataset.nClasses=3755;
  char filenameFormat[]="Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for(int fileNumber=1001;fileNumber<=1240;fileNumber++) {
    sprintf(filename,filenameFormat,fileNumber);
    readPotFile(dataset.pictures,filename);
  }
  calculateRegularizingConstants(dataset);
  return dataset;
}
SpatialDataset Casia101112TrainSet() {
  SpatialDataset dataset;
  dataset.name="CASIA OLHWDB1.1 train set--240 writers";
  dataset.type=TRAINBATCH;
  dataset.nFeatures=nInputFeatures;
  dataset.nClasses=3755;
  char filenameFormatA[]="Data/CASIA_pot_files/%03d.pot";
  char filenameFormatB[]="Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for(int fileNumber=1;fileNumber<=420;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readPotFile(dataset.pictures,filename);
  }
  for(int fileNumber=501;fileNumber<=800;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readPotFile(dataset.pictures,filename);
  }
  for(int fileNumber=1001;fileNumber<=1240;fileNumber++) {
    sprintf(filename,filenameFormatB,fileNumber);
    readPotFile(dataset.pictures,filename);
  }
  calculateRegularizingConstants(dataset);
  return dataset;
}
SpatialDataset Casia11TestSet() {
  SpatialDataset dataset;
  dataset.name="CASIA OLHWDB1.1 test set--60 writers";
  dataset.type=TESTBATCH;
  dataset.nFeatures=nInputFeatures;
  dataset.nClasses=3755;
  char filenameFormat[]="Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for(int fileNumber=1241;fileNumber<=1300;fileNumber++) {
    sprintf(filename,filenameFormat,fileNumber);
    readPotFile(dataset.pictures,filename);
  }
  return dataset;
}
