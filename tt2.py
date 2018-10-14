from yoloOpencv import opencvYOLO
import cv2
import imutils
import time, math

yolo = opencvYOLO(modeltype="yolov3", objnames="../darknet/data/coco.names",
    weights="../darknet/weights/yolov3.weights", cfg="../darknet/cfg/yolov3.cfg")

calculateLine1 = [(350, 570), (560, 570)]  #Down [(from(x,y), to(x,y)]
calculateLine2 = [(510, 640), (800, 640)]  #Up [(from(x,y), to(x,y)]
calculateLine3 = [(810, 490), (1020, 490)]  #Up [(from(x,y), to(x,y)]
calculateRange_y = 60   # length of Y (for up or down of the line)

video_file = "tt2.mp4"
output_video = "/media/sf_ShareFolder/tw_highway/tt2.avi"

def draw_CalculateLine(frame):
    cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]), (calculateLine1[1][0],calculateLine1[1][1]), (0, 0, 255), 2)
    cv2.line(frame, (calculateLine2[0][0],calculateLine2[1][1]), (calculateLine2[1][0],calculateLine2[1][1]), (255, 0, 0), 2)
    cv2.line(frame, (calculateLine3[0][0],calculateLine3[1][1]), (calculateLine3[1][0],calculateLine3[1][1]), (255, 0, 0), 2)
    return frame

def bbox2Centroid(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    return (int(x+(w/2)), int(y+(h/2)))

def in_range_S2N(bbox):
    #only calculate the cars (from south to north) run across the line and not over Y +- calculateRange_y
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    cx = int(x+(w/2))
    cy = int(y+(h/2))

    if(cx>=calculateLine1[0][0] and cx<=calculateLine1[1][0] and \
        cy<=calculateLine1[0][1]+calculateRange_y and cy>(calculateLine1[1][1]-calculateRange_y) ):
        return True

    else:
        return False

def in_range_N2S(bbox):
    #only calculate the cars (from south to north) run across the line and not over Y +- calculateRange_y
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    cx = int(x+(w/2))
    cy = int(y+(h/2))

    if(cx>=calculateLine2[0][0] and cx<=calculateLine2[1][0] and \
        cy>calculateLine2[0][1]-calculateRange_y and cy<=(calculateLine2[1][1]+calculateRange_y) ):
        return True

    else:
        return False

def in_range_N2S_2(bbox):
    #only calculate the cars (from south to north) run across the line and not over Y +- calculateRange_y
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    cx = int(x+(w/2))
    cy = int(y+(h/2))

    if(cx>=calculateLine3[0][0] and cx<=calculateLine3[1][0] and \
        cy>calculateLine3[0][1]-calculateRange_y and cy<=(calculateLine3[1][1]+calculateRange_y) ):
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

def printText(img, carNum1, truckNum1, busNum1, othersNum1,  carNum2, truckNum2, busNum2, othersNum2, carNum3, truckNum3, busNum3, othersNum3):
    y1 = 30
    y2 = 55
    y3 = 80
    y4 = 105

    cv2.putText(img, "Car:", (490, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, "Truck:", (490, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, "Bus:", (490, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, "Others:", (490, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, str(carNum2), (600, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(truckNum2), (600, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(busNum2), (600, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(othersNum2), (600, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, "Car:", (690, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Truck:", (690, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Bus:", (690, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Others:", (690, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, str(carNum1), (800, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(truckNum1), (800, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(busNum1), (800, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(othersNum1), (800, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, "Car:", (880, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Truck:", (880, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Bus:", (880, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Others:", (880, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, str(carNum3), (990, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(truckNum3), (990, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(busNum3), (990, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(othersNum3), (990, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

#ob jects informations for the last frame
count_Car1 = 0
count_Truck1 = 0
count_Bus1 = 0
count_Others1 = 0
count_Car2 = 0
count_Truck2 = 0
count_Bus2 = 0
count_Others2 = 0
count_Car3 = 0
count_Truck3 = 0
count_Bus3 = 0
count_Others3 = 0

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
    #width = 728
    #height = 656
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video,fourcc, 30.0, (int(width),int(height)))


    frameID = 0
    while True:
        print("FrameID=", frameID)

        hasFrame, frame = VIDEO_IN.read()
        frameCopy = frame.copy()

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

        yolo.getObject(frameCopy, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        cv2.imshow("TEST", imutils.resize(frameCopy, width=600))

        #Get the objects only in the calculate range
        for id, labelName in enumerate(yolo.nms_labelNames):
            box = yolo.nms_bboxes[id]
            score = yolo.nms_scores[id]
            if((in_range_S2N(box)==True or in_range_N2S(box)==True or in_range_N2S_2(box)==True) and box[3]<300):
                if(labelName!="car" and labelName!="bus" and labelName!="truck"):
                    labelName = "others"

                now_IDs.append(id)
                now_BBOXES.append(box)
                now_CENTROIDS.append(bbox2Centroid(box))
                now_LABELS.append(labelName)
                now_COUNTED.append(False)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
                cv2.putText(frame, labelName, (box[0]+10, box[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        print("B: last:{}, now:{}".format(last_CENTROIDS, now_CENTROIDS))

        UP = False
        UP2 = False
        DOWN = False

        if(frameID>0 and len(now_CENTROIDS)>0):
            obj_target = count_Object(last_CENTROIDS, now_CENTROIDS)
            print("last:{}, now:{}".format(last_CENTROIDS, now_CENTROIDS))
            print("OBJ_TARGETS:", obj_target)

            for id, now_id in enumerate(obj_target):
                #if last Y is under the line and now Y is above or on the line, then count += 1
                #print(last_CENTROIDS[id][1], calculateLine2[0][1], now_CENTROIDS[now_id][1], calculateLine2[0][1])

                if (last_CENTROIDS[id][1]>calculateLine2[0][1]) and (now_CENTROIDS[now_id][1]<=calculateLine2[0][1]):
                    UP = True
                if (last_CENTROIDS[id][1]>calculateLine3[0][1]) and (now_CENTROIDS[now_id][1]<=calculateLine3[0][1]):
                    UP2 = True
                if (last_CENTROIDS[id][1]<calculateLine1[0][1]) and (now_CENTROIDS[now_id][1]>=calculateLine1[0][1]):
                    DOWN = True

                print(DOWN, UP, UP2)
                if( UP is True):
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck1 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car1 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus1 += 1
                    else:
                        count_Others1 += 1

                elif( UP2 is True):
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck3 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car3 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus3 += 1
                    else:
                        count_Others3 += 1

                elif( DOWN is True):
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck2 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car2 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus2 += 1
                    else:
                        count_Others2 += 1


        frame = draw_CalculateLine(frame)
        frame = printText(frame, count_Car1, count_Truck1, count_Bus1, count_Others1, count_Car2, count_Truck2, count_Bus2, count_Others2, \
            count_Car3, count_Truck3, count_Bus3, count_Others3)

        cv2.imshow("Frame", imutils.resize(frame, width=600))
        out.write(frame)

        frameID += 1

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
