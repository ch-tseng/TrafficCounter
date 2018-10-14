from yoloOpencv import opencvYOLO
import cv2
import imutils
import time, math

yolo = opencvYOLO(modeltype="yolov3", objnames="../darknet/data/coco.names",
    weights="../darknet/weights/yolov3.weights", cfg="../darknet/cfg/yolov3.cfg")

calculateLine1 = [(460, 580), (1194, 552)]  #[(from(x,y), to(x,y)]

calculateRange_x = 30   # length of X (for up or down of the line)
calculateRange_y = 30   # length of Y (for up or down of the line)

video_file = "/media/sf_ShareFolder/hw/city.mp4"
output_video = "/media/sf_ShareFolder/city.avi"

def draw_CalculateLine(frame, lineboder=40):
    cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]+20), (calculateLine1[1][0],calculateLine1[1][1]+25), (0, 255, 0), lineboder)
    cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]), (calculateLine1[1][0],calculateLine1[1][1]), (0, 0, 255), lineboder)


    #cv2.imshow("TEST", frame)
    return frame

def bbox2Centroid(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    return (int(x+(w/2)), int(y+(h/2)))

def in_range(img, bbox):
    #only calculate the cars (from south to north) run across the line and not over Y +- calculateRange_y
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    cx = int(x+(w/2))
    cy = int(y+(h/2))

    if(img[cy,cx][0]==0 and img[cy,cx][1]==0 and img[cy,cx][2]==255) or (img[cy,cx][0]==0 and img[cy,cx][1]==255 and img[cy,cx][2]==0):
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

def printText(img, num1_1, num1_2):
    y1 = 70
    y2 = 150
    y3 = 230
    y4 = 310
    y_add = 840
    fontSize = 2.0
    fontcolor = (0,255,255)

    cv2.putText(img, "Car:", (60, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, "Truck:", (60, y2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, "Bus:", (60, y3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, "Others:", (60, y4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, str(num1_1[0]), (300, y1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_1[1]), (300, y2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_1[2]), (300, y3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_1[3]), (300, y4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)

    cv2.putText(img, "Car:", (1160, y1+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, "Truck:", (1160, y2+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, "Bus:", (1160, y3+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, "Others:", (1160, y4+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontcolor, 3)
    cv2.putText(img, str(num1_2[0]), (1400, y1+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_2[1]), (1400, y2+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_2[2]), (1400, y3+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)
    cv2.putText(img, str(num1_2[3]), (1400, y4+y_add), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 255, 0), 3)

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
        frameLayout = draw_CalculateLine(frameLayout, 40)

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
            if(in_range(frameLayout, box)==True):
                now_IDs.append(id)
                now_BBOXES.append(box)
                now_CENTROIDS.append(bbox2Centroid(box))
                now_LABELS.append(labelName)
                now_COUNTED.append(False)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
                if(labelName=="motorbike" or labelName=="person"):
                    labelName = "others"

                cv2.putText(frame, labelName, (box[0]+30, box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)
            #frame = cv2.circle(frame, ( int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2))), 6, (0,0,255), -1)
        #print("B: last:{}, now:{}".format(last_CENTROIDS, now_CENTROIDS))

        if(frameID>0 and len(now_CENTROIDS)>0):
            obj_target = count_Object(last_CENTROIDS, now_CENTROIDS)
            #print("last:{}, now:{}".format(last_CENTROIDS, now_CENTROIDS))
            #print("OBJ_TARGETS:", obj_target)

            UP_1 = False
            DOWN_1 = False

            for id, now_id in enumerate(obj_target):
                #if last Y is under the line and now Y is above or on the line, then count += 1
                #print(frame[now_CENTROIDS[now_id][1],now_CENTROIDS[now_id][0]], frame[last_CENTROIDS[id][1],last_CENTROIDS[id][0]])
                color_now = frameLayout[now_CENTROIDS[now_id][1],now_CENTROIDS[now_id][0]]
                color_last = frameLayout[last_CENTROIDS[id][1],last_CENTROIDS[id][0]]

                UP_1 = (color_now == [0,0,255]).all() and (color_last == [0,255,0]).all() 
                DOWN_1 = (color_last == [0,0,255]).all() and (color_now==[0,255,0]).all()
                print("LINE1", UP_1, DOWN_1)

                if( UP_1 == True):
                    print("UP_1 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck1_1 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car1_1 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus1_1 += 1
                    else:
                        count_Motorbike1_1 += 1

                if( DOWN_1 == True):
                    print("DOWN_1 add!!")
                    if(now_LABELS[now_id]=="truck"):
                        count_Truck1_2 += 1
                    elif(now_LABELS[now_id]=="car"):
                        count_Car1_2 += 1
                    elif(now_LABELS[now_id]=="bus"):
                        count_Bus1_2 += 1
                    else:
                        count_Motorbike1_2 += 1

                if( (UP_1 == True) or (DOWN_1 == True)):
                    cv2.rectangle(frame, (now_BBOXES[now_id][0], now_BBOXES[now_id][1]),\
                        (now_BBOXES[now_id][0]+now_BBOXES[now_id][2], now_BBOXES[now_id][1]+now_BBOXES[now_id][3]), (0,0,255), 2)

        num1_1 = (count_Car1_1, count_Truck1_1, count_Bus1_1, count_Motorbike1_1)
        num1_2 = (count_Car1_2, count_Truck1_2, count_Bus1_2, count_Motorbike1_2)
        frame = printText(frame, num1_1, num1_2)

        cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]), (calculateLine1[1][0],calculateLine1[1][1]), (0, 255, 0), 2)
        cv2.imshow("Frame", imutils.resize(frame, width=850))
        out.write(frame)

        frameID += 1

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
