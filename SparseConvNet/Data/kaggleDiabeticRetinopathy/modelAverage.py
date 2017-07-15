##This is probably rather more complicated than needed.

import numpy, scipy, cPickle
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LinearRegression as lr

fl=[ "kaggleDiabeticRetinopathyCompetitionModelFiles/kaggleDiabetes1_epoch65",
     "kaggleDiabeticRetinopathyCompetitionModelFiles/kaggleDiabetes2_epoch84",
     "kaggleDiabeticRetinopathyCompetitionModelFiles/kaggleDiabetes3_epoch46",
       ]

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
for l in open("sd_Train"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in trainData["left"]:
        trainData[b[1]][b[0]].append(float(a[1]))
for l in open("noise_Train"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in trainData["left"]:
        trainData[b[1]][b[0]].append(float(a[1]))
for l in open("meanColor_Size_Train"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in trainData["left"]:
        trainData[b[1]][b[0]]+=[float (x) for x in a[1:]]
trainN=sorted(trainClasses["left"].keys())
trainX=numpy.array([trainData["left"][n]+trainData["right"][n] for n in trainN]+
                   [trainData["right"][n]+trainData["left"][n] for n in trainN])
trainY=numpy.array([trainClasses["left"][n] for n in trainN]+
                   [trainClasses["right"][n] for n in trainN])
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
for l in open("sd_Train"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in validationData["left"]:
        validationData[b[1]][b[0]].append(float(a[1]))
for l in open("noise_Train"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in validationData["left"]:
        validationData[b[1]][b[0]].append(float(a[1]))
for l in open("meanColor_Size_Train"):
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

fitForest=False
if fitForest:
    r=rf(1000,n_jobs=5)
    r.fit(trainX,trainY)
    pickleSave(r,"kaggleDiabeticRetinopathyCompetitionModelFiles/modelAverage.forest.pickle")
else:
    r=pickleLoad("kaggleDiabeticRetinopathyCompetitionModelFiles/modelAverage.forest.pickle")
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
for l in open("sd_Test"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    testData[b[1]][b[0]].append(float(a[1]))
for l in open("noise_Test"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    testData[b[1]][b[0]].append(float(a[1]))
for l in open("meanColor_Size_Test"):
    a=l.split(" ")
    b=a[0].split(".")[0].split("_")
    if b[0] in testData[b[1]]:
        testData[b[1]][b[0]]+=[float (x) for x in a[1:]]
testData["right"]["25313"]=testData["left"]["25313"] ##Cases where conversion failed as no radius could be calculated
testData["right"]["27096"]=testData["left"]["27096"]
testN=sorted(testData["left"].keys())
testX=numpy.array([testData["left"][n]+testData["right"][n] for n in testN]+[testData["right"][n]+testData["left"][n] for n in testN])
testZ=[n+"_left" for n in testN]+[n+"_right" for n in testN]
#########################################################################################
print "Test data"
a=(r.predict_proba(testX)*numpy.arange(5).reshape((1,5))).sum(1)
threshold=numpy.array([ 0.56923337,  1.3709647,   2.30468216,  3.11985482])
testY=(a>threshold[0])*1+(a>threshold[1])*1+(a>threshold[2])*1+(a>threshold[3])*1
print "testY bincount:", numpy.bincount(testY, minlength=5)
f=open("submissionFile","w")
print >>f, "image,level"
for x in zip(testZ,testY):
    print >>f, "%s,%d"%x
