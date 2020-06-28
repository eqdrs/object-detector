from __future__ import print_function
from sklearn.feature_extraction.image import extract_patches_2d
from pipeline.object_detection import helpers
from pipeline.descriptors import HOG
from pipeline.utils import dataset
from pipeline.utils import Conf
from imutils import paths
from scipy import io
import numpy as np
import progressbar
import argparse
import random
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "Diretório do arquivo de configuração")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

hog = HOG(orientations = conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
    cellsPerBlock = tuple(conf["cells_per_block"]), normalize = conf["normalize"])
data = []
labels = []

trnPaths = list(paths.list_images(conf["image_dataset"]))
trnPaths = random.sample(trnPaths, int(len(trnPaths) * conf["percent_gt_images"]))
print("Descrevendo ROIs para o treinamento...")

widgets = ["Extraindo: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(trnPaths), widgets = widgets).start()

for (i, trnPath) in enumerate(trnPaths):
    image = cv2.imread(trnPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageID = trnPath[trnPath.rfind("_") + 1:].replace(".jpg", "")

    p = "{}/annotation_{}.mat".format(conf["image_annotations"], imageID)
    bb = io.loadmat(p)["box_coord"][0]
    roi = helpers.crop_ct101_bb(image, bb, padding = conf["offset"], dstSize = tuple(conf["sliding_window_dim"]))

    rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)

    for roi in rois:
        features = hog.describe(roi)
        data.append(features)
        labels.append(1)

    pbar.update(i)

pbar.finish()
dstPaths = list(paths.list_images(conf["image_distractions"]))
pbar = progressbar.ProgressBar(maxval = conf["num_distraction_images"], widgets = widgets).start()
print("Descrevendo ROIs das imagens negativas...")

for i in np.arange(0, conf["num_distraction_images"]):
    image = cv2.imread(random.choice(dstPaths))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patches = extract_patches_2d(image, tuple(conf["sliding_window_dim"]),
        max_patches = conf["num_distraction_per_image"])

    for patch in patches:
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)

    pbar.update(i)

pbar.finish()
print("Gravando features e labels no arquivo...")
dataset.dump_dataset(data, labels, conf["features_path"], "features")
