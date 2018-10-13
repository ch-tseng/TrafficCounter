from yoloOpencv import opencvYOLO
import cv2
import imutils
import time, math

yolo = opencvYOLO(modeltype="yolov3", objnames="../darknet/data/coco.names",
    weights="../darknet/weights/yolov3.weights", cfg="../darknet/cfg/yolov3.cfg")

calculateLine1 = [(36, 610), (1126, 900)]  #[(from(x,y), to(x,y)]
calculateLine2 = [(820, 350), (1574, 530)]  #[(from(x,y), to(x,y)]
calculateLine3 = [(830, 380), (0, 580)]  #[(from(x,y), to(x,y)]
calculateLine4 = [(1532, 590), (1224, 880)]  #[(from(x,y), to(x,y)]

calculateRange_x = 30   # length of X (for up or down of the line)
calculateRange_y = 30   # length of Y (for up or down of the line)

video_file = "/media/sf_ShareFolder/traffic_taiachun.mp4"
output_video = "/media/sf_ShareFolder/traffic2.avi"

def draw_CalculateLine(frame):
    cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]+20), (calculateLine1[1][0],calculateLine1[1][1]+25), (0, 255, 0), 20)
    cv2.line(frame, (calculateLine2[0][0],calculateLine2[0][1]-20), (calculateLine2[1][0],calculateLine2[1][1]-25), (0, 255, 0), 20)
    cv2.line(frame, (calculateLine3[0][0],calculateLine3[0][1]-20), (calculateLine3[1][0],calculateLine3[1][1]-25), (0, 255, 0), 20)
    cv2.line(frame, (calculateLine4[0][0],calculateLine4[0][1]+25), (calculateLine4[1][0],calculateLine4[1][1]+35), (0, 255, 0), 25)
    cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]), (calculateLine1[1][0],calculateLine1[1][1]), (0, 0, 255), 30)
    cv2.line(frame, (calculateLine2[0][0],calculateLine2[0][1]), (calculateLine2[1][0],calculateLine2[1][1]), (0, 0, 254), 30)
    cv2.line(frame, (calculateLine3[0][0],calculateLine3[0][1]), (calculateLine3[1][0],calculateLine3[1][1]), (0, 0, 253), 30)
    cv2.line(frame, (calculateLine4[0][0],calculateLine4[0][1]), (calculateLine4[1][0],calculateLine4[1][1]), (0, 0, 252), 30)


    #cv2.imshow("TEST", frame)
    return frame

def bbox2Centroid(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    return (int(x+(w/2)), int(y+(h/2)))

def in_range(img, bbox, line):
    #only calculate the cars (from south to north) run across the line and not over Y +- calculateRange_y
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    cx = int(x+(w/2))
    cy = int(y+(h/2))

    if(line ==1):
        if(img[cy,cx][0]==0 and img[cy,cx][1]==0 and img[cy,cx][2]==255) or (img[cy,cx][0]==0 and img[cy,cx][1]==255 and img[cy,cx][2]==0):
            return True
        else:
            return False
    elif(line ==2):
        if(img[cy,cx][0]==0 and img[cy,cx][1]==0 and img[cy,cx][2]==254) or (img[cy,cx][0]==0 and img[cy,cx][1]==255 and img[cy,cx][2]==0):
            return True
        else:
            return False
    elif(line ==3):
        if(img[cy,cx][0]==0 and img[cy,cx][1]==0 and img[cy,cx][2]==253) or (img[cy,cx][0]==0 and img[cy,cx][1]==255 and img[cy,cx][2]==0):
            return True
        else:
            return False
    elif(line ==4):
        if(img[cy,cx][0]==0 and img[cy,cx][1]==0 and img[cy,cx][2]==252) or (img[cy,cx][0]==0 and img[cy,cx][1]==255 and img[cy,cx][2]==0):
            return True
        else:
            return False


def distance(p1, p2):
    dx2 = (p1[0] - p2[0])**2          # (200-10)^2
    dy2 = (p1[1] - p2[1])**2          # (300-20)^2
    distance = math.sqrt(dx2 + dy2)
    #print(p1[0], p2[0], p1[1], p2[1], dx2, dy2, distance)

    return distance

def count_Object(centroid_last, centroid_now):
    distances = []

    for cent_last in centroid_last:
        smallist = 99999.0
        smallist_id = 0
        dist = 0.0
	
        for id, cent_now in enumerate(centroid_now):
            dist = distance(cent_now, cent_last)
            if(dist<=smallist):
                smallist = dist
                smallist_id = id

        distances.append(smallist_id)

    return distances

def printText(img, num1_1, num1_2, num2_1, num2_2, num3_1, num3_2, num4_1, num4_2):
    y1 = 50
    y2 = 100
    y3 = 150
    y4 = 200
    fontSize = 1.5

    cv2.putText(img, "Car:", (1530, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, "Truck:", (1530, y2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, "Bus:", (1530, y3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, "Motorbike:", (1530, y4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, str(num2_1[0])+"/"+str(num2_2[0]), (1790, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num2_1[1])+"/"+str(num2_2[1]), (1790, y2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num2_1[2])+"/"+str(num2_2[2]), (1790, y3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num2_1[3])+"/"+str(num2_2[3]), (1790, y4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)

    cv2.putText(img, "Car:", (30, y1+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, "Truck:", (30, y2+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, "Bus:", (30, y3+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, "Motorbike:", (30, y4+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), 3)
    cv2.putText(img, str(num1_1[0])+"/"+str(num1_2[0]), (260, y1+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_1[1])+"/"+str(num1_2[1]), (260, y2+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_1[2])+"/"+str(num1_2[2]), (260, y3+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_1[3])+"/"+str(num1_2[3]), (260, y4+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)

    cv2.putText(img, "Car:", (60, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 3)
    cv2.putText(img, "Truck:", (60, y2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 3)
    cv2.putText(img, "Bus:", (60, y3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 3)
    cv2.putText(img, "Motorbike:", (60, y4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 3)
    cv2.putText(img, str(num3_1[0])+"/"+str(num3_2[0]), (320, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num3_1[1])+"/"+str(num3_2[1]), (320, y2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num3_1[2])+"/"+str(num3_2[2]), (320, y3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num3_1[3])+"/"+str(num3_2[3]), (320, y4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)

    cv2.putText(img, "Car:", (1530, y1+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 2)
    cv2.putText(img, "Truck:", (1530, y2+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 2)
    cv2.putText(img, "Bus:", (1530, y3+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 2)
    cv2.putText(img, "Motorbike:", (1530, y4+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 255, 255), 2)
    cv2.putText(img, str(num4_1[0])+"/"+str(num4_2[0]), (1790, y1+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num4_1[1])+"/"+str(num4_2[1]), (1790, y2+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num4_1[2])+"/"+str(num4_2[2]), (1790, y3+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num4_1[3])+"/"+str(num4_2[3]), (1790, y4+830), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)


    return img

#ob jects informations for the last frame
count_Car1_1 = 0
count_Truck1_1 = 0
count_Bus1_1 = 0
count_Motorbike1_1 = 0
count_Car1_2 = 0
count_Truck1_2 = 0
count_Bus1_2 = 0
count_Motorbike1_2 = 0

count_Car2_1 = 0
count_Truck2_1 = 0
count_Bus2_1 = 0
count_Motorbike2_1 = 0
count_Car2_2 = 0
count_Truck2_2 = 0
count_Bus2_2 = 0
count_Motorbike2_2 = 0

count_Car3_1 = 0
count_Truck3_1 = 0
count_Bus3_1 = 0
count_Motorbike3_1 = 0
count_Car3_2 = 0
count_Truck3_2 = 0
count_Bus3_2 = 0
count_Motorbike3_2 = 0

count_Car4_1 = 0
count_Truck4_1 = 0
count_Bus4_1 = 0
count_Motorbike4_1 = 0
count_Car4_2 = 0
count_Truck4_2 = 0
count_Bus4_2 = 0
count_Motorbike4_2 = 0


last_IDs = []
last_BBOXES = []
last_CENTROIDS = []
last_LABELS = []

now_IDs = []
now_BBOXES = []
now_CENTROIDS = []
now_LABELS = []
now_COUNTED = []

start_time = time.time()

if __name__ == "__main__":

    VIDEO_IN = cv2.VideoCapture(video_file)
    # Get current width of frame
    width = VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video,fourcc, 30.0, (int(width),int(height)))


    frameID = 0
    while True:
        print("FrameID=", frameID)

        hasFrame, frame = VIDEO_IN.read()
        frameLayout = frame.copy()
        frameLayout = draw_CalculateLine(frameLayout)

        last_IDs = now_IDs.copy()
        last_BBOXES = now_BBOXES.copy()
        last_CENTROIDS = now_CENTROIDS.copy()
        last_LABELS = now_LABELS.copy()

        now_IDs.clear()
        now_BBOXES.clear()
        now_CENTROIDS.clear()
        now_LABELS.clear()
        now_COUNTED.clear()

        #print("A: last:{}, now:{}".format(last_CENTROIDS, now_CENTROIDS))
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("--- %s seconds ---" % (time.time() - start_time))
            break

        yolo.getObject(frame, labelWant="", drawBox=False, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))

        #Get the objects only in the calculate range
        for id, labelName in enumerate(yolo.nms_labelNames):
            box = yolo.nms_bboxes[id]
            score = yolo.nms_scores[id]
            if(in_range(frameLayout, box, 1)==True or in_range(frameLayout, box, 2)==True or in_range(frameLayout, box, 3)==True or in_range(frameLayout, box, 4)==True):
                now_IDs.append(id)
                now_BBOXES.append(box)
                now_CENTROIDS.append(bbox2Centroid(box))
                now_LABELS.append(labelName)
                now_COUNTED.append(False)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)

            #frame = cv2.circle(frame, ( int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2))), 6, (0,0,255), -1)
        #print("B: last:{}, now:{}".format(last_CENTROIDS, now_CENTROIDS))

        if(frameID>0 and len(now_CENTROIDS)>0):
            obj_target = count_Object(last_CENTROIDS, now_CENTROIDS)
            #print("last:{}, now:{}".format(last_CENTROIDS, now_CENTROIDS))
            #print("OBJ_TARGETS:", obj_target)

            UP_1 = False
            DOWN_1 = False
            UP_2 = False
            DOWN_2 = False
            UP_3 = False
            DOWN_3 = False
            UP_4 = False
            DOWN_4 = False

            for id, now_id in enumerate(obj_target):
                #if last Y is under the line and now Y is above or on the line, then count += 1
                #print(frame[now_CENTROIDS[now_id][1],now_CENTROIDS[now_id][0]], frame[last_CENTROIDS[id][1],last_CENTROIDS[id][0]])
                color_now = frameLayout[now_CENTROIDS[now_id][1],now_CENTROIDS[now_id][0]]
                color_last = frameLayout[last_CENTROIDS[id][1],last_CENTROIDS[id][0]]

                UP_1 = (color_now == [0,0,255]).all() and (color_last == [0,255,0]).all() 
                DOWN_1 = (color_last == [0,0,255]).all() and (color_now==[0,255,0]).all()
                print("LINE1", UP_1, DOWN_1)
                UP_2 = (color_now == [0,0,254]).all() and (color_last == [0,255,0]).all() 
                DOWN_2 = (color_last == [0,0,254]).all() and (color_now==[0,255,0]).all()
                print("LINE2", UP_2, DOWN_2)
                UP_3 = (color_now == [0,0,253]).all() and (color_last == [0,255,0]).all() 
                DOWN_3 = (color_last == [0,0,253]).all() and (color_now==[0,255,0]).all()
                print("LINE3", UP_3, DOWN_3)
                UP_4 = (color_now == [0,0,252]).all() and (color_last == [0,255,0]).all() 
                DOWN_4 = (color_last == [0,0,252]).all() and (color_now==[0,255,0]).all()
                print("LINE4", UP_4, DOWN_4)

                if( UP_1 == True):
                    print("UP_1 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck1_1 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car1_1 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus1_1 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike1_1 += 1

                if( DOWN_1 == True):
                    print("DOWN_1 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck1_2 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car1_2 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus1_2 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike1_2 += 1

                if( UP_2 == True):
                    print("UP_2 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck2_1 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car2_1 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus2_1 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike2_1 += 1

                if( DOWN_2 == True):
                    print("DOWN_2 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck2_2 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car2_2 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus2_2 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike2_2 += 1

                if( UP_3 == True):
                    print("UP_3 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck3_1 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car3_1 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus3_1 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike3_1 += 1

                if( DOWN_3 == True):
                    print("DOWN_2 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck3_2 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car3_2 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus3_2 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike3_2 += 1

                if( UP_4 == True):
                    print("UP_4 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck4_1 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car4_1 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus4_1 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike4_1 += 1

                if( DOWN_4 == True):
                    print("DOWN_4 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck4_2 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car4_2 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus4_2 += 1
                    elif(now_LABELS[now_id]=="motorbike"):
                        count_Motorbike4_2 += 1


                if( (UP_1 == True) or (DOWN_1 == True) or (UP_2 == True) or (DOWN_2 == True) or (UP_3 == True) or (DOWN_3 == True) or (UP_4 == True) or (DOWN_4 == True)):
                    cv2.rectangle(frame, (now_BBOXES[now_id][0], now_BBOXES[now_id][1]),\
                        (now_BBOXES[now_id][0]+now_BBOXES[now_id][2], now_BBOXES[now_id][1]+now_BBOXES[now_id][3]), (0,0,255), 2)

        num1_1 = (count_Car1_1, count_Truck1_1, count_Bus1_1, count_Motorbike1_1)
        num1_2 = (count_Car1_2, count_Truck1_2, count_Bus1_2, count_Motorbike1_2)
        num2_1 = (count_Car2_1, count_Truck2_1, count_Bus2_1, count_Motorbike2_1)
        num2_2 = (count_Car2_2, count_Truck2_2, count_Bus2_2, count_Motorbike2_2)
        num3_1 = (count_Car3_1, count_Truck3_1, count_Bus3_1, count_Motorbike3_1)
        num3_2 = (count_Car3_2, count_Truck3_2, count_Bus3_2, count_Motorbike3_2)
        num4_1 = (count_Car4_1, count_Truck4_1, count_Bus4_1, count_Motorbike4_1)
        num4_2 = (count_Car4_2, count_Truck4_2, count_Bus4_2, count_Motorbike4_2)
        frame = printText(frame, num1_1, num1_2, num2_1, num2_2, num3_1, num3_2, num4_1, num4_2)

        cv2.imshow("Frame", imutils.resize(frame, width=850))
        out.write(frame)

        frameID += 1

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
