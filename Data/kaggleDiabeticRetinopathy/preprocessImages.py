#Preprocess training images.
#Scale 300 seems to be sufficient; 500 and 1000 are overkill
import cv2, glob, numpy

def scaleRadius(img,scale):
    x=img[img.shape[0]/2,:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

for scale in [300, 500]:
    for f in (glob.glob("train/*.jpeg")+glob.glob("test/*.jpeg"))[2::3]:
        try:
            a=cv2.imread(f)
            a=scaleRadius(a,scale)
            b=numpy.zeros(a.shape)
            cv2.circle(b,(a.shape[1]/2,a.shape[0]/2),int(scale*0.9),(1,1,1),-1,8,0)
            aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
            cv2.imwrite(str(scale)+"_"+f,aa)
        except:
            print f
