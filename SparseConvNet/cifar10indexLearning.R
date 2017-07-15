a=as.matrix(read.table("CIFAR-10 train set.features"))
trainX=a[,-(1:2)]
#trainX=trainX*3*(trainX<0)+trainX*(trainX>0)
trainY=a[,2]
a=as.matrix(read.table("CIFAR-10 test set.features"))
testX=a[,-(1:2)]
#testX=testX*3*(testX<0)+testX*(testX>0)
testY=a[,2]
rm(a)
library(e1071)
s=svm(trainX[1:1000,],trainY[1:1000],type="C",kernel="lin",cost=1)
mean(predict(s,trainX)==trainY)
mean(predict(s,testX)==testY)

library(grid)
load("/home/ben/Archive/Datasets/cifar/cifar-10-R/cifar10.RData")
for (i in 1:dim(trainX)[2]) {
  a=train.X[order(trainX[,i])[c(1:25,40000+9976:10000)],]/255
  #a=test.X[order(testX[,i])[c(1:25,9976:10000)],]/255
  a=array(aperm(array(a,c(5,10,32,32,3)),c(4,1,3,2,5)),c(32*5,32*10,3))
  png(paste("index-learning/",i,".png",sep=""),height=32*5,width=32*10)
  grid.raster(a,interpolate=F)
  dev.off()
}

par(mfrow=c(2,5))
for (i in 0:9) {
  a=cor(trainX,trainY==i)
  b=cor(testX,testY==i)
  plot(a[order(a)])
  points(b[order(a)],col=2)
}

