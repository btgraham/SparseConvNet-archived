#include "NetworkArchitectures.h"
#include "SpatiallySparseDatasetKagglePlankton.h"

int epoch=0;
int cudaDevice=-1; // PCI bus ID: -1 for default GPU
int batchSize=50; // Increase/decrease according to GPU memory
#define VALIDATION

Picture* OpenCVPicture::distort(RNG& rng, batchType type) {
  OpenCVPicture* pic=new OpenCVPicture(*this);
  pic->loadData();
  float c00=1, c01=0;
  float c10=0, c11=1;
  c00=c11=100.0/scale*rng.uniform(0.8,1.2); //scale diagonal to length 100 pixels
  if (rng.randint(2)==0) c00*=-1; //Mirror image
  {float alpha=rng.uniform(-3.14159265,3.14159265);
    matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha));} //Rotate
  if (type==TRAINBATCH) {
    int r=rng.randint(2);
    if (r==0) {
      float alpha=rng.uniform(-0.2,0.2);
      matrixMul2x2inPlace(c00,c01,c10,c11,1,0,alpha,1); //Slant
    }
    if (r==1) {
      float alpha=rng.uniform(-0.1,0.1);
      matrixMul2x2inPlace(c00,c01,c10,c11,1-alpha,0,0,1+alpha); //Stretch
    }
  }
  if (type==TRAINBATCH) {
    float alpha=rng.uniform(-3.14159265,3.14159265);
    matrixMul2x2inPlace(c00,c01,c10,c11,cos(alpha),-sin(alpha),sin(alpha),cos(alpha)); //Rotate
  }
  transformImage(pic->mat, backgroundColor, c00, c01, c10, c11);
  pic->centerMass();
  if (type==TRAINBATCH)
    pic->jiggle(rng,25);
  return pic;
}

int main() {
  std::string baseName="weights/kagglePlankton";

  SpatiallySparseDataset trainSet=KagglePlanktonTrainSet();
  trainSet.summary();
#ifdef VALIDATION
  SpatiallySparseDataset valSet=trainSet.extractValidationSet();
  trainSet.summary();
  valSet.summary();
#endif

  DeepCNet cnn(2,6,32,VLEAKYRELU,trainSet.nFeatures,trainSet.nClasses,0.5f,cudaDevice);

  if (epoch>0) {
    cnn.loadWeights(baseName,epoch);
  }
  for (epoch++;epoch<=500;epoch++) {
    std::cout <<"epoch: " << epoch << std::endl;
    cnn.processDataset(trainSet, batchSize,0.003*exp(-epoch*0.01));
    if (epoch%10==0) {
      cnn.saveWeights(baseName,epoch);
#ifdef VALIDATION
      cnn.processDatasetRepeatTest(valSet, batchSize/2, 3);
#endif
    }
  }
  SpatiallySparseDataset testSet=KagglePlanktonTestSet();
  testSet.summary();
  cnn.processDatasetRepeatTest(testSet, batchSize/2, 12,"submission_1.csv","image,acantharia_protist,acantharia_protist_big_center,acantharia_protist_halo,amphipods,appendicularian_fritillaridae,append,cularian_slight_curve,appendicularian_s_shape,appendicularian_straight,artifacts,artifacts_edge,chaetognath_non_sagitta,chaet,gnath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_cala,oid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large,copepod_calanoid_large_side_antennatucked,copepod_calano,d_octomoms,copepod_calanoid_small_longantennae,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona,copepod_cyclopoid_oithona_,ggs,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophor,_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_plu,eus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larv,_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,eupha,siids,euphausiids_young,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_mycto,hids,fish_larvae_thin_body,fish_larvae_very_thin_body,header,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,h,dromedusae_h15,hydromedusae_haliscera,hydromedusae_haliscera_small_sideview,hydromedusae_liriope,hydromedusae_narco_dark,hydr,medusae_narcomedusae,hydromedusae_narco_young,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA,hydromedusae_s,apeA_sideview_small,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae,typeD,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_la,vae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp-like_,ther,shrimp_sergestidae,shrimp_zoea,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophor,_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes,siphonophore_calycophoran_sphaeronectes_stem,siphonoph,re_calycophoran_sphaeronectes_young,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect,siphonophore_physone,t_young,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tu,t,trochophore_larvae,tunicate_doliolid,tunicate_doliolid_nurse,tunicate_partial,tunicate_salp,tunicate_salp_chains,unknown_bl,bs_and_smudges,unknown_sticks,unknown_unclassified\n");
}
