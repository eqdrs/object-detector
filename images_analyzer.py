from __future__ import print_function
from pipeline.utils import Conf
from scipy import io
import numpy as np
import argparse
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "Diretório do arquivo de configuração")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
widths = []
heights = []

for p in glob.glob(conf["image_annotations"] + "/*.mat"):
    (y, h, x, w) = io.loadmat(p)["box_coord"][0]
    widths.append(w - x)
    heights.append(h - y)

(avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
print()
print()
print("Largura média: {:.2f}".format(avgWidth))
print("Altura média: {:.2f}".format(avgHeight))
print("Aspect ratio: {:.2f}".format(avgWidth / avgHeight))
print()
print()
