import numpy, cv2, itertools
cls=['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering', 'HammerThrow', 'HandStandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']


for a in [#'trainlist01'
          #,
          'testlist01_'
          ]:
    outF=file(a+'.dataset','w')
    for l in file('ucfTrainTestlist/'+a+'.txt'):
        f=l.split(" ")[0].split("/")[1].split("\r\n")[0]
        cl=cls.index(l.split("_")[1])
        print f, cl,
        cap=cv2.VideoCapture('videos/'+f)
        nFrames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print nFrames,
        frames=[]
        for i in range(nFrames):
            r=cap.read()
            if r[0]:
                frames.append(cv2.resize(r[1],(160,120)))
        outF.write(f+' '+str(cl))
        data=[]
        for q in range(len(frames)-1):
            diff=frames[q+1]*1.0-frames[q]*1.0
            zz=abs(diff).sum(2)>100
            diff=(diff*(zz[:,:,None])).astype("int32")
            zzzx,zzzy=numpy.nonzero(zz)
            for j in range(zzzx.size):
                x=zzzx[j]
                y=zzzy[j]
                data.extend([x,y,q,diff[x,y,0],diff[x,y,1],diff[x,y,2]])
        print len(data),
        print "!",
        if len(data)==0:
            print "!!!!!!!!!!"
        data.insert(0,len(data)/6)
        for d in data:
            outF.write(' '+str(d))
        outF.write('\n')
        print "!"
    outF.close()
