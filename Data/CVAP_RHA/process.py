import numpy, cv2
for a in ['train', 'test','validation'
            ]:
    outF=file(a+'.dataset',"w")
    for l in file(a+".txt"):
        ll=l.split(" ")
        ll[-1]=ll[-1].split('\r\n')[0]
        f=ll[0]
        cl=['boxing','handclapping','handwaving','jogging','running','walking'].index(ll[0].split("_")[1])
        print ll
        cap=cv2.VideoCapture('videos/'+f+'_uncomp.avi')
        r=cap.read()
        frames=[]
        while r[0]:
            frames.append(r[1][:,:,0])
            r=cap.read()
        while len(frames)<int(ll[-1]):
              frames.append(numpy.zeros((120,160)))
              print "*",
        for i in range(1,len(ll)/2):
            start=int(ll[2*i])-1
            stop=int(ll[2*i+1])-1
            outF.write(f+' '+str(cl)+' ')
            data=[]
            for q in range(start,stop):
                diff=frames[q+1]*1.0-frames[q]*1.0
                diff=diff*(abs(diff)>30)
                for x in range(120):
                    for y in range(160):
                        if diff[x,y]!=0:
                            data.extend([x,y,q-start,int(diff[x,y])])
            outF.write(str(len(data)/4)+' '+reduce(lambda i,j: i+" "+j,[str(s) for s in data])+'\n')
    outF.close()
