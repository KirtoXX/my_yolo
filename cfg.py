import numpy as np

NORM_H, NORM_W = 416, 416
GRID_H, GRID_W = 13 , 13
BATCH_SIZE = 4
BOX = 5  #total box of output is 13*13*5=845
ORIG_CLASS = 3
CLASS = 3
IMG_INPUT_SIZE = 416

LABEL_FILE = 'data/ILSVRC/synset_words_2.txt'

THRESHOLD = 0.2
# If a feature extractor performed 5 max-pooling --> Image resolution being reduced 2^5 = 32 times
# Most current state-of-the-art models have max-pooling layers (August, 2017)
SHRINK_FACTOR  = 32

#ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
ANCHORS = '2.926972,4.099791,  0.893706,1.049178,  5.172482,7.103885,  0.392892,0.608273,  1.613534,2.047487'

ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]

SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0
