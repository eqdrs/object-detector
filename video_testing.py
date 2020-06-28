from pipeline.object_detection import non_max_suppression
from pipeline.object_detection import bb_intersection_over_union
from pipeline.object_detection import ObjectDetector
from pipeline.descriptors import HOG
from pipeline.utils import Conf
import numpy as np
import imutils
import argparse
import pickle
import streamlit as st
import cv2

from scipy import io
import glob
import pdb

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "Diretório do arquivo de configuração")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations = conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
	cellsPerBlock = tuple(conf["cells_per_block"]), normalize = conf["normalize"], block_norm = "L1")
od = ObjectDetector(model, hog)

path = conf["video_path"]

cap = cv2.VideoCapture(path) 

while (cap.isOpened()):

  ret, image = cap.read()
  image = imutils.resize(image, width = min(260, image.shape[1]))
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  (boxes, probs) = od.detect(gray, conf["sliding_window_dim"], winStep = conf["window_step"],
    pyramidScale = conf["pyramid_scale"], minProb = conf["min_probability"])
  pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
  orig = image.copy()

  for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

  cv2.imshow("Video", image)
  if cv2.waitKey(1) == 27: 
    break

cv2.destroyAllWindows()
