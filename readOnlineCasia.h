//Obtain POT files from http://www.nlpr.ia.ac.cn/databases/handwriting/home.html and put them in a directory CASIA_pot_files/
//const int nCharacters = 3755;

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


int readPotFile(vector<Picture*> &characters, const char* filename, bool numberLabelsFromZero = true) {
  ifstream file(filename,ios::in|ios::binary);
  if (!file) {
    cout <<"Cannot find " << filename << endl;
    exit(EXIT_FAILURE);}
  potCharacterHeader pCH;
  iPoint iP;
  FloatPoint fP;
  while (file.read((char*)&pCH,sizeof(potCharacterHeader))) {
    OnlinePicture* character = new OnlinePicture;
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
    if (numberLabelsFromZero) {
      character->label=find(gbcodesPOT,gbcodesPOT+3755,character->label)-gbcodesPOT;
      if (character->label<nCharacters)
        characters.push_back(character);
      else
        delete character;
    } else {
      characters.push_back(character);
    }
  }
  file.close();
  return 0;
}
#ifdef CASIASMALL
void loadData()
{
  char filenameFormat[]="Data/CASIA_pot_files/1%03d.pot";
  char filename[100];
  for(int fileNumber=10;fileNumber<=100;fileNumber+=10) {
    sprintf(filename,filenameFormat,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=5;fileNumber<=100;fileNumber+=10) {
    sprintf(filename,filenameFormat,fileNumber);
    readPotFile(testCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  cout << endl;
}
#endif
#ifdef CASIA10
void loadData()
{
  char filenameFormat[]="Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for(int fileNumber=1241;fileNumber<=1300;fileNumber++) {
    sprintf(filename,filenameFormat,fileNumber);
    readPotFile(testCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=1001;fileNumber<=1240;fileNumber++) {
    sprintf(filename,filenameFormat,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  cout << endl;
}
#endif
#ifdef CASIA101112
void loadData()
{
  char filenameFormatA[]="Data/CASIA_pot_files/%03d.pot";
  char filenameFormatB[]="Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for(int fileNumber=1241;fileNumber<=1300;fileNumber++) {
    sprintf(filename,filenameFormatB,fileNumber);
    readPotFile(testCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=1;fileNumber<=420;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=501;fileNumber<=800;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=1001;fileNumber<=1240;fileNumber++) {
    sprintf(filename,filenameFormatB,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  cout << endl;
}
#endif

#ifdef CASIA101112_small_test_set
void loadData()
{
  char filenameFormatA[]="Data/CASIA_pot_files/%03d.pot";
  char filenameFormatB[]="Data/CASIA_pot_files/%04d.pot";
  char filename[100];
  for(int fileNumber=1242;fileNumber<=1300;fileNumber+=10) {
    sprintf(filename,filenameFormatB,fileNumber);
    readPotFile(testCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=1;fileNumber<=420;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=501;fileNumber<=800;fileNumber++) {
    sprintf(filename,filenameFormatA,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  for(int fileNumber=1001;fileNumber<=1240;fileNumber++) {
    sprintf(filename,filenameFormatB,fileNumber);
    readPotFile(trainCharacters,filename);
    cout << "\r" << filename << " " <<trainCharacters.size()<< " " << testCharacters.size();
  }
  cout << endl;
}
#endif
