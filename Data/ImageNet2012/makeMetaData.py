# # Need
# # ILSVRC2012_bbox_train_v2/
# # ILSVRC2012_img_train/
# # ILSVRC2012_img_test/
# # ILSVRC2012_img_val/
# # ILSVRC2012_devkit_t12/
# # ILSVRC2014_clsloc_validation_blacklist.txt
# # 2012_clsloc_classnames.txt


import csv
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET



trainingData=file("trainingData.txt","w")
for x in csv.reader(file("2012_clsloc_classnames.txt"),delimiter=" "):
    print x
    x[0]=int(x[0])
    files = [ f for f in listdir(join("ILSVRC2012_img_train",x[1])) if isfile(join("ILSVRC2012_img_train",x[1],f)) and f[0]=='n']
    for f in files:
        f1=join("ILSVRC2012_img_train",x[1],f)
        # f2=join("ILSVRC2012_bbox_train_v2",x[1],f[:-5]+".xml")
        # if isfile(f2):
        #     tree = ET.parse(f2)
        #     xsize=str(tree.find("size").find("width").text)
        #     ysize=str(tree.find("size").find("height").text)
        #     for obj,zzzzz in zip(tree.iterfind("object"),[1]):
        #         print >>trainingData, x[0], f1, xsize, ysize, int(obj.find("bndbox").find("xmin").text), int(obj.find("bndbox").find("xmax").text), int(obj.find("bndbox").find("ymin").text), int(obj.find("bndbox").find("ymax").text)
        # else:
        print >>trainingData, x[0], f1, 0, 0, 0, 0, 0, 0
trainingData.close()

validationData=file("validationData.txt","w")
blacklist=csv.reader(file("ILSVRC2014_clsloc_validation_blacklist.txt"))
blacklist= [int(x[0]) for x in blacklist]
for x,c in zip(csv.reader(file("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")),range(1,50001)):
    if not c in blacklist:
        f1="ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG"%c
        print >>validationData, x[0], f1, 0, 0, 0, 0, 0, 0
validationData.close()


testData=file("testData.txt","w")
for c in range(1,100001):
    f1="ILSVRC2012_img_test/ILSVRC2012_test_%08d.JPEG"%c
    print >>testData, c, f1, 0, 0, 0, 0, 0, 0
testData.close()
