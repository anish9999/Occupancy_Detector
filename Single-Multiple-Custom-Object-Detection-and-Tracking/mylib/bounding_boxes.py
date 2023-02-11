import numpy as np
from yolov3_tf2.utils import convert_boxes
from deep_sort.detection import Detection
from deep_sort import preprocessing

class BoundingBox:
    
    def  __init__ (self, metric):
        self.metric = metric
        self.detections = []

        

    def update(self, img, img_in, classes, names, scores, boxes, class_names, encoder):
        
        nms_max_overlap = 0.8
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
        return self.detections
        
    
    