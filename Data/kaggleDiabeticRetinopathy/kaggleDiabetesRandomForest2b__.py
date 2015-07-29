import numpy, scipy, cPickle
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LinearRegression as lr

fl=[# "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes101_epoch69", #0.816 0.456
    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes300_epoch127", #0.838 0.373  (numbers are for accuracy, mse, isolated)
    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes301_epoch266", #0.846 0.322
     "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes302_epoch84", #0.853 0.276
    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes305_epoch97", #0.852 0.309
    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes305NiN_epoch52", #0.853 0.32
     "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes307_epoch65", #0.864 0.288
    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes309_epoch140", #0.848 0.295
     "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes1006_epoch46", #0.859 0.285

    #  "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes377_epoch121", #0.848 0.31
    #  "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes501_epoch148", #0.851 0.316
    #  "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes502_epoch116", #0.854 0.298
    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes308_epoch65", #0.851 0.302

    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes307b_epoch80", #0.853 0.303
    # "/home/ben/Desktop/Code/SparseConvNet/dr/kaggleDiabetes307c_epoch78", #0.86 0.296

       ]

# ?1? ?5? ?303
#########################################################################################
trainData={"left": {}, "right": {}}
trainClasses={"left": {}, "right": {}}

def list_add(x,y):
    return [a+b for a,b in zip(x,y)]

for f in fl:
    for l in open(f+".train"):
        a=l.split("/")[-1].split(",")
        b=a[0].split(".")[0].split("_")
        trainClasses[b[1]][b[0]]=int(a[1])
        a=[float(x) for x in a[2:]]
        if b[0] not in trainData[b[1]]:
            trainData[b[1]][b[0]]=a
        else:
            trainData[b[1]][b[0]]=list_add(trainData[b[1]][b[0]],a)
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/___train/sd"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in trainData["left"]:
        trainData[b[1]][b[0]].append(float(a[1]))
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/___train/noise"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in trainData["left"]:
        trainData[b[1]][b[0]].append(float(a[1]))
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/meanColor_Size_Train"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in trainData["left"]:
        trainData[b[1]][b[0]]+=[float (x) for x in a[1:]]
trainN=sorted(trainClasses["left"].keys())
trainX=numpy.array([trainData["left"][n]+trainData["right"][n] for n in trainN]+[trainData["right"][n]+trainData["left"][n] for n in trainN])
trainY=numpy.array([trainClasses["left"][n] for n in trainN]+[trainClasses["right"][n] for n in trainN])
trainZ=[n+"_left" for n in trainN]+[n+"_right" for n in trainN]
#########################################################################################
validationData={"left": {}, "right": {}}
validationClasses={"left": {}, "right": {}}

for f in fl:
    for l in open(f+".validation"):
        a=l.split("/")[-1].split(",")
        b=a[0].split(".")[0].split("_")
        validationClasses[b[1]][b[0]]=int(a[1])
        a=[float(x) for x in a[2:]]
        if b[0] not in validationData[b[1]]:
            validationData[b[1]][b[0]]=a
        else:
            validationData[b[1]][b[0]]=list_add(validationData[b[1]][b[0]],a)
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/___train/sd"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in validationData["left"]:
        validationData[b[1]][b[0]].append(float(a[1]))
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/___train/noise"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in validationData["left"]:
        validationData[b[1]][b[0]].append(float(a[1]))
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/meanColor_Size_Train"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in validationData["left"]:
        validationData[b[1]][b[0]]+=[float (x) for x in a[1:]]
validationN=sorted(validationClasses["left"].keys())
validationX=numpy.array([validationData["left"][n]+validationData["right"][n] for n in validationN]+[validationData["right"][n]+validationData["left"][n] for n in validationN])
validationY=numpy.array([validationClasses["left"][n] for n in validationN]+[validationClasses["right"][n] for n in validationN])
validationZ=[n+"_left" for n in validationN]+[n+"_right" for n in validationN]
#########################################################################################
def pickleSave(obj,name):
    pkl_file = open(name, 'wb')
    cPickle.dump(obj,pkl_file,protocol=-1)
    pkl_file.close()

def pickleLoad(name):
    pkl_file = open(name, 'rb')
    obj=cPickle.load(pkl_file)
    pkl_file.close()
    return obj

print "Validation data"

#r=rf(1000,n_jobs=5)
#r.fit(trainX,trainY)
#pickleSave(r,"kaggleDiabetesRandomForest2b__.forest.pickle")
r=pickleLoad("kaggleDiabetesRandomForest2b__.forest.pickle")
print "accuracy", numpy.mean(r.predict(validationX)==validationY)
print "mse", numpy.mean((r.predict(validationX)-validationY)**2)
a=(r.predict_proba(validationX)*numpy.arange(5).reshape((1,5))).sum(1)

#########################################################################################
testData={"left": {}, "right": {}}

for f in fl:
    for l in open(f+".test"):
        a=l.split("/")[-1].split(",")
        b=a[0].split(".")[0].split("_")
        a=[float(x) for x in a[1:]]
        if b[0] not in testData[b[1]]:
            testData[b[1]][b[0]]=a
        else:
            testData[b[1]][b[0]]=list_add(testData[b[1]][b[0]],a)
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/___test/sd"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    testData[b[1]][b[0]].append(float(a[1]))
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/___test/noise"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    testData[b[1]][b[0]].append(float(a[1]))
for l in open("/home/ben/Archive/Datasets/kaggleDiabeticRetinopathy/meanColor_Size_Test"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in testData[b[1]]:
        testData[b[1]][b[0]]+=[float (x) for x in a[1:]]
testData["right"]["25313"]=testData["left"]["25313"]
testData["right"]["27096"]=testData["left"]["27096"]
testN=sorted(testData["left"].keys())
testX=numpy.array([testData["left"][n]+testData["right"][n] for n in testN]+[testData["right"][n]+testData["left"][n] for n in testN])
testZ=[n+"_left" for n in testN]+[n+"_right" for n in testN]
#########################################################################################
print "Test data"
a=(r.predict_proba(testX)*numpy.arange(5).reshape((1,5))).sum(1)

threshold=numpy.array([ 0.56923337,  1.3709647,   2.30468216,  3.07985482+0.04])
#threshold=numpy.array([0.67354101,  1.36130863,  2.20032235,  2.90541836])
testY=(a>threshold[0])*1+(a>threshold[1])*1+(a>threshold[2])*1+(a>threshold[3])*1
print "testY bincount:", numpy.bincount(testY, minlength=5)
f=open("test.predictions_2b__","w")
print >>f, "image,level"
for x in zip(testZ,testY):
    print >>f, "%s,%d"%x
