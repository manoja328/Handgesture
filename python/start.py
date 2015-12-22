__author__ = 'manoj'
import cv
import math
import cv2
import numpy as np
import csv
import ctypes
MOUSE_LEFTDOWN = 0x0002     # left button down
MOUSE_LEFTUP = 0x0004       # left button up
MOUSE_RIGHTDOWN = 0x0008    # right button down
MOUSE_RIGHTUP = 0x0010      # right button up
MOUSE_MIDDLEDOWN = 0x0020   # middle button down
MOUSE_MIDDLEUP = 0x0040     # middle button up
slidemove=0
textbelow=['one','Show V','three','Rock on!','punch you!','palm open','Cick Mouse']

input_layer_size=900  # M * N of input image
hidden_layer_size=25
num_labels=7  #no of class to recognize
Theta1 = np.zeros([hidden_layer_size,input_layer_size+1])
Theta2 = np.zeros([num_labels,hidden_layer_size+1])

def make_formatted_csv():
    first="Theta1.csv"
    second="Theta2.csv"
    global Theta1,Theta2
    i=0
    for line in csv.reader(open(first)):
        for x in range(input_layer_size+1):
            Theta1[i,x]=float(line[x])
        i+=1

    #-----------------------
    i=0
    for line in csv.reader(open(second)):
        for x in range(hidden_layer_size+1):
            Theta2[i,x]=float(line[x])
        i+=1



def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def predict(X):
    global Theta1,Theta2
    m = X.shape[0]
    #print X
    #print X.shape
    #num_labels = Theta2.shape[0]
    #p = np.zeros([X.shape[0],1])


    #Theta1 = np.matrix(np.random.rand(hidden_layer_size,input_layer_size+1))
    #Theta2 = np.matrix(np.random.rand(num_labels,hidden_layer_size+1))

    #a=np.ones([56,2])
    #np.hstack((np.ones([a.shape[0],1]),a))

    h1=sigmoid(np.matrix(np.hstack((np.ones([m,1]),X))) * Theta1.transpose())
    h2=sigmoid(np.matrix(np.hstack((np.ones([m,1]),h1))) * Theta2.transpose())

    pred= h2.argmax(1)
    return (pred+1)


make_formatted_csv()

def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()


myfont = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 3, cv.CV_AA)
(fstx, fsty)=(0,0)
INIT_TIME = 10
B_PARAM = 1.0 / 30.0
zeta = 10.0
capture = cv.CaptureFromCAM(1)
_red = (0, 0, 255, 0)
_green = (0, 255, 0, 0)
image = cv.CreateImage((640, 480), 8, 3)
img1 = cv.CreateImage((640, 480), 8, 3)
av = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_32F, 3)
sgm = cv.CreateImage(cv.GetSize(image), 32, 3)
lower = cv.CreateImage(cv.GetSize(image), 32, 3)
upper = cv.CreateImage(cv.GetSize(image), 32, 3)
tmp = cv.CreateImage(cv.GetSize(image), 32, 3)
dst = cv.CreateImage(cv.GetSize(image), 8, 3)
msk = cv.CreateImage(cv.GetSize(image), 8, 1)
skin = cv.CreateImage(cv.GetSize(image), 8, 1)
skin = cv.CreateImage(cv.GetSize(image), 8, 1)
handlarge = cv.CreateImage(cv.GetSize(image), 8, 1)
handorig = cv.CreateImage(cv.GetSize(image), 8, 3)
copyskin = cv.CreateImage(cv.GetSize(image), 8, 1)
firstcopyskin = cv.CreateImage(cv.GetSize(image), 8, 1)
circleimg = cv.CreateImage((640, 480), 8, 1)
destblobimg = cv.CreateImage((640, 480), 8, 1)
drawimage = cv.CreateImage(cv.GetSize(image), cv.IPL_DEPTH_8U, 3)
savehandimage = cv.CreateImage((30, 30), cv.IPL_DEPTH_8U, 1)
cv.SetZero(av)
for i in range(0, INIT_TIME):
    image = cv.QueryFrame(capture)
    cv.Flip(image, image, 1)
    cv.Acc(image, av)
cv.ConvertScale(av, av, 1.0 / INIT_TIME)
cv.SetZero(sgm)
for i in range(0, INIT_TIME):
    image = cv.QueryFrame(capture)
    cv.Flip(image, image, 1)
    cv.Convert(image, tmp)
    cv.Sub(tmp, av, tmp)
    cv.Pow(tmp, tmp, 2.0)
    cv.ConvertScale(tmp, tmp, 2.0)
    cv.Pow(tmp, tmp, 0.5)
    cv.Acc(tmp, sgm)

cv.ConvertScale(sgm, sgm, 1.0 / INIT_TIME)
prevfst=(0,0)
while True:
    image = cv.QueryFrame(capture)
    cv.Flip(image, image, 1)

    cv.Convert(image, tmp)
    cv.Sub(av, sgm, lower)
    cv.SubS(lower, cv.ScalarAll(zeta), lower)
    cv.Add(av, sgm, upper)
    cv.AddS(upper, cv.ScalarAll(zeta), upper)
    cv.InRange(tmp, lower, upper, msk)

    cv.Sub(tmp, av, tmp)
    cv.Pow(tmp, tmp, 2.0)
    cv.ConvertScale(tmp, tmp, 2.0)
    cv.Pow(tmp, tmp, 0.5)
    cv.RunningAvg(image, av, B_PARAM, msk)
    cv.RunningAvg(tmp, sgm, B_PARAM, msk)

    cv.Not(msk, msk)
    cv.RunningAvg(tmp, sgm, 0.01, msk)
    cv.Copy(msk, skin)

    cv.Erode(skin, skin, None, 1)
    cv.Erode(skin, skin, None, 1)
    cv.Dilate(skin, skin, None, 1)
    cv.Dilate(skin, skin, None, 1)

    storage = cv.CreateMemStorage(0)
    firstcopyskin = cv.CloneImage(skin)
    contour = cv.FindContours(firstcopyskin, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE, (0, 0))
    blobarea = 0
    (rectx, rexty, rectw, recth) = (0, 0, 0, 0)
    for c in contour_iterator(contour):
        PointArray2D32f = cv.CreateMat(1, len(c), cv.CV_32FC2)
        for (i, (x, y)) in enumerate(c):
            PointArray2D32f[0, i] = (x, y)

        # Draw the current contour in gray
        #gray = cv.CV_RGB(100, 100, 100)
        #cv.DrawContours(image, c, _red ,_green,10)
        #cv.DrawContours(image,c,cv.ScalarAll(255),cv.ScalarAll(255),100,2,8)
        if cv.ContourArea(c) > blobarea:
            blobarea = cv.ContourArea(c)
            rectx, rexty, rectw, recth = cv.BoundingRect(c)
            #print rect#x,y,w,h

    if rectw > 0 or recth > 0:
        if recth > 190:
            recth = 190
            #cv.SetImageROI(image,(rectx,rexty,rectw,recth))
        #cv.ShowImage("Source", image)
        #cv.ResetImageROI(image)
        #else:
        #    cv.ShowImage("Source", image)


        hand = (rectx, rexty, rectw, recth)
        cv.Zero(handlarge)
        cv.Zero(handorig)
        cv.SetImageROI(skin, hand)
        cv.SetImageROI(handlarge, hand)

        cv.Copy(skin, handlarge)
        cv.ResetImageROI(skin)
        cv.ResetImageROI(handlarge)
        cv.Dilate(handlarge, handlarge, None, 1)
        cv.Erode(handlarge, handlarge, None, 1)

        cv.Copy(image, handorig, handlarge)
        #cvDrawRect(image,cvPoint(hand.x,hand.y),cvPoint(hand.x + hand.width , hand.y+hand.height),CV_RGB(255,0,0))

        copyskin = cv.CloneImage(handlarge)

        moments = cv.Moments(handlarge)
        try:
            mom10 = cv.GetSpatialMoment(moments, 1, 0)
            mom01 = cv.GetSpatialMoment(moments, 0, 1)
            area = cv.GetCentralMoment(moments, 0, 0)
            posX = int(mom10 / area)
            posY = int(mom01 / area)
            #cv.Circle(image,(posX,posY),3,cv.CV_RGB(0,255,0), 3, 8, 0 )

        except:
            pass

        contour = cv.FindContours(copyskin, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE, (0, 0))
        blobarea = 0
        larrect = (0, 0, 0, 0)
        larseq = 0
        for c in contour_iterator(contour):
            PointArray2D32f = cv.CreateMat(1, len(c), cv.CV_32FC2)
            for (i, (x, y)) in enumerate(c):
                PointArray2D32f[0, i] = (x, y)
            if cv.ContourArea(c) > blobarea:
                blobarea = cv.ContourArea(c)
                larrect = cv.BoundingRect(c)
                larseq = c

        if larrect[2] > 0 or larrect[3] > 0:
            cv.Rectangle(image, (larrect[0], larrect[1]), (larrect[0] + larrect[2], larrect[1] + larrect[3]),
                         cv.CV_RGB(255, 0, 0), 2)
            cv.DrawContours(image, larseq, cv.ScalarAll(255), cv.ScalarAll(255), 100, 2, 8)
            cv.DrawContours(handorig, larseq, cv.CV_RGB(255, 0, 0), cv.ScalarAll(255), 255)
            hull_storage = cv.CreateMemStorage(0)
            storage3 = cv.CreateMemStorage(0)
            hulls = cv.ConvexHull2(larseq, hull_storage, cv.CV_CLOCKWISE, 0)
            defectarray = cv.ConvexityDefects(larseq, hulls, storage3)
            farthestdistance = 0

            for defect in defectarray:
                start = defect[0]
                end = defect[1]
                depth = defect[2]
                ab = start
                dist = math.sqrt((float)(posX - ab[0]) * (posX - ab[0]) + (posY - ab[1]) * (posY - ab[1]))

                if (dist > farthestdistance and posY > ab[1]):
                    farthestdistance = dist
                    fstx = ab[0]
                    fsty = ab[1]
                cv.Line(image, start, depth, cv.CV_RGB(255, 255, 0), 1, cv.CV_AA, 0)
                cv.Circle(image, depth, 5, cv.CV_RGB(0, 255, 0), 2, 8, 0)#green
                cv.Circle(image, start, 5, cv.CV_RGB(0, 0, 255), 2, 8, 0)#blue
                cv.Line(image, depth, end, cv.CV_RGB(255, 255, 0), 1, cv.CV_AA, 0)




        fst = (fstx, fsty)
        #for hand drawing
        cv.Line(drawimage,fst,prevfst ,cv.CV_RGB(0,0,255),5)
        cv.Circle(drawimage, fst, 5, cv.CV_RGB(0,0,255), -2, 8,10)
        prevfst=fst
        cv.ShowImage("drawing image",drawimage)

        cv.Circle(image, ( posX, posY ), 5, cv.CV_RGB(0, 255, 0), 5)
        cv.Circle(handorig, ( posX, posY ), 5, cv.CV_RGB(0, 255, 0), 5)
        rad = math.sqrt((float)(posX - fst[0]) * (posX - fst[0]) + (posY - fst[1]) * (posY - fst[1]))
        cv.Line(image, fst, ( posX, posY ), cv.CV_RGB(255, 255, 0), 4)
        cv.Circle(image, ( posX, posY ), int(rad / 1.5), cv.CV_RGB(255, 255, 255), 6)
        cv.Line(handorig, fst, ( posX, posY ), cv.CV_RGB(255, 255, 0), 4)
        cv.Circle(handorig, fst, 5, cv.CV_RGB(0, 0, 255), 5)
        #nos of fingers


        cv.Zero(circleimg)
        cv.Zero(destblobimg)
        cv.Circle(circleimg, (posX, posY ), int(rad / 1.5), cv.CV_RGB(255, 255, 255), 6)
        cv.And(handlarge, circleimg, destblobimg)
        #cv.ShowImage("circleimg",circleimg)
        #cv.ShowImage("blobimg",destblobimg)
        contour = cv.FindContours(destblobimg, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE, (0, 0))
        fingnocnt = 0
        for c in contour_iterator(contour):
            fingnocnt += 1
        if fingnocnt >= 1:
            lab = str(fingnocnt - 1)
        else:
            lab = str(0)
        cv.PutText(image, lab, (10, 50), myfont, cv.CV_RGB(255, 0, 0))

    ##        #save the hand
    ##        if (fingnocnt == 2):
    ##            imageno=imageno+1
    ##
    ##            #imagename=str(fingnocnt-1)+"/ges"+str(imageno)+".bmp"
    ##            #imagename="5/ges" +str(imageno)+".bmp"
    ##
    ##            imagename="ges" +str(imageno)+"d.bmp"
    ##            cv.SetImageROI(handlarge,larrect)
    ##            cv.Resize(handlarge,savehandimage)
    ##            cv.Dilate(savehandimage,savehandimage,0,1)
    ##            cv.Erode(savehandimage,savehandimage,0,1)
    ##            cv.SaveImage(imagename,savehandimage)
    ##            cv.ResetImageROI(handlarge)

        cv.SetImageROI(handlarge,larrect)
        cv.Resize(handlarge,savehandimage)
        #cv.Dilate(savehandimage,savehandimage,None,1)
        #cv.Erode(savehandimage,savehandimage,None,1)
        cv.SaveImage("my.bmp",savehandimage)
        cv.ResetImageROI(handlarge)
        cvimage = cv2.imread("my.bmp",cv2.CV_LOAD_IMAGE_GRAYSCALE)
        b=np.matrix(cvimage,np.float64,True)
        a=np.reshape(b,[1,b.shape[0]*b.shape[1]],order='F')
        astd=np.std(a,1) + 0.0000001
        d=(a-np.mean(a,1))/astd
        pred=predict(d)

        #predtext="Using Neural Network: "+str(pred)
        predtext="Using Neural Network: "+textbelow[pred-1]
        cv.PutText(image,predtext,(10, 440), myfont, cv.CV_RGB(255, 0, 0))
        if pred==4:
            cv.SetZero(drawimage)

        if pred==5:
            cv.SaveImage("dynamic.bmp",drawimage)
            cv.PutText(image,"image saved",(300, 50), myfont, cv.CV_RGB(255, 0, 0))
            slidemove+=1
            if slidemove % 30 ==0:
                    #ctypes.windll.user32.SetCursorPos(500,500)
                    #ctypes.windll.user32.mouse_event(MOUSE_LEFTDOWN ,0,0,0,0)
                    #ctypes.windll.user32.mouse_event(MOUSE_LEFTUP,0,0,0,0)
                    slidemove=0



    cv.ShowImage("largehand only-binary color", handlarge)
    cv.ShowImage("largehand only-original color", handorig)
    cv.ShowImage("original image", image)
    cv.ShowImage("all foreground", skin)
    key = cv.WaitKey(1)
    if key == 27:
        break

cv.DestroyAllWindows()
del capture


