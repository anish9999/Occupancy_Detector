from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2, itertools
import matplotlib.pyplot as plt
from itertools import zip_longest
import pandas as pd
from mylib import config

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# main.py
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
from collections import OrderedDict

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
# class_names = [c.strip() for c in open('./data/labels/coco_copy.names').readlines()]
# class_names = ['person']
# print(c.strip() for c in open('./data/labels/coco_copy.names').readlines())
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8
trackableObjects = {}
trackers = []
W = None
H = None


# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
x = []
empty=[]
empty1=[]
rects = []
n = []
arrs = OrderedDict()

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)


model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)
vid = cv2.VideoCapture('.//data/video/example_01.mp4')
# vid = VideoStrea/m(config.url).start()
# time.sleep(2.0)

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = 4
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
# vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_width,vid_height = 500, 600
out = cv2.VideoWriter('./data/video/output_6.avi', codec, vid_fps, (vid_width, vid_height))

from collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

counter = []
Inside_Counter = []
Outside_Counter = []
a = []

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

while True:
    _,img = vid.read()
    # img = imutils.resize(img, width = 500)
    
    if W is None or H is None:
        (H, W) = img.shape[:2]
    if img is None:
        print('Completed')
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
    current_count = int(0)



    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)
  
		
        # objects = ct.update(rects)
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)
        
        
        # add the bounding box coordinates to the rectangles list
        startX = int(bbox[0])
        startY = int(bbox[1])
        endX   = int(bbox[2])
        endY   = int(bbox[3])
        rects.append((startX, startY, endX, endY))

        
        # use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
        # objects = ct.update(rects)
        centroid = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            centroid[i] = (cX, cY)
            objectID = track.track_id
            arrs[objectID] = centroid[i]       
    
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

  
        for (objectID, centroid) in arrs.items():

            to = trackableObjects.get(objectID, None)

            if to is not None:
              
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)


                if not to.counted:
                    # print(centroid[1])
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        empty.append(totalUp)
                        to.counted = True
                        					
                    elif direction > 0 and centroid[1] > H // 2:
                        if class_name == 'person' or class_name == 'dog':
                            totalDown += 1
                            empty1.append(totalDown)
                            x = []					
                            x.append(len(set(empty1))-len(set(empty)))
                            to.counted = True      
            else:
                to = TrackableObject(objectID, centroid)
                # print(to.centroids)
                a = a + (to.centroids)
        
        
        
		# store the trackable object in our dictionary
        trackableObjects[objectID] = to

        y = {}
        for k in range(0, len((a))):
            print((a))
            y[objectID] = a[k]
            for (objectID, centroid) in y.items():
                b = centroid
        r = b[1]
        # print(r)
        
        center_y = int(((bbox[1])+(bbox[3]))/2)
        # print(center_y)
        if center_y <= int(3*H/6+H/20) and center_y >= int(3*H/6-H/20):
            if class_name == 'person' or class_name == 'dog':
                counter.append(int(track.track_id))
                current_count += 1
        elif center_y >= int(3*H/6+H/20) and center_y >= r:
            if class_name == 'person' or class_name == 'dog':
                Inside_Counter.append(int(track.track_id))
        elif center_y <= int(3*H/6-H/20) and center_y < r:
            if class_name == 'person' or class_name == 'dog':
                Outside_Counter.append(int(track.track_id))        
                        
    info = [
	("Exit", totalUp),
	("Enter", totalDown),
	]
        
    info2 = [
    ("Total people inside", x),
	]

        # Display the output
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(img, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    for (i, (k, v)) in enumerate(info2):
        text = "{}: {}".format(k, v)
        cv2.putText(img, text, ((W - 250), H - ((i * 10) + 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        
    cv2.putText(img, "Current People Count: " + str(current_count), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
    total_count = [] 
    total_count.append(len(set(counter)))
    inside_count =[]  
    outside_count = []  

    
    inside_count.append(set(Inside_Counter))
    outside_count.append(set(Outside_Counter))
    # print(inside_count)
    # print(outside_count)
    # Max_Inside_count.append()
    cv2.putText(img, "Total People Count: " + str(total_count), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        # draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
    cv2.line(img, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
    cv2.putText(img, "-Prediction border - Entrance-", (15, H - ((i * 20) + 194)),
	cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


    height, width, _ = img.shape
    cv2.line(img, (0, int(3*H/6+H/20)), (W, int(3*H/6+H/20)), (0, 255, 0), thickness=2)
    cv2.line(img, (0, int(3*H/6-H/20)), (W, int(3*H/6-H/20)), (0, 255, 0), thickness=2)
       

    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    cv2.imshow('output', img)
    # cv2.resizeWindow('output', 600, 600)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()