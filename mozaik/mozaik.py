#!/usr/bin/python

import sys, getopt
import matplotlib.pyplot as plt
import os

from primitive import *
from method import *

def loadImage(filename):
    return cv2.imread(filename)

def main(argv):
    inputfile = ''
    outputfile = ''
    shapesCount = 200
    primitive = RectanglePrimitive
    man = 'mozaik.py -i <inputfile> [-o <outputfile> -c <shapeCounts> -p <primitive>]'
    try:
        opts, args = getopt.getopt(argv,"hi:o:c:p:",["input=","output=","count=","primitive="])
    except getopt.GetoptError:
        print man
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print man
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
        elif opt in ("-c", "--count"):
            shapesCount = int(arg)
        elif opt in ("-p", "--primitive"):
            if arg == "rectangle":
                primitive = RectanglePrimitive
            elif arg == "triangle":
                primitive = TrianglePrimitive
            elif arg == "ellipse":
                primitive = EllipsePrimitive
            elif arg == "circle":
                primitive = CirclePrimitive
            elif arg == "square":
                primitive = SquarePrimitive
            else:
                print 'Primitive not recognized'
                sys.exit(2)
    if not inputfile:
        print man
        sys.exit(2)
    else:
        random.seed()
        img = loadImage(inputfile)
        if img is None:
            print man
            sys.exit(2)
        # result = randomGeneration(img, shapesCount, primitive, maxSize=256, randomIterations=4000)
        # result = annealingGeneration(img, shapesCount, primitive, maxSize=256, randomIterations=400, T0=30, Tf=0.01, tau=0.99)
        # result = hillClimbGeneration(img, shapesCount, primitive, maxSize=256, randomIterations=400, randomPopSelection=0.10, hillClimbIterations=40)
        result = gradientGeneration(img, shapesCount, primitive, maxSize=256, randomPopSelection=0.25, randomIterations=100)


        if not outputfile:
            outputfile = '{}_{}_{:1.0f}.png'.format(os.path.splitext(os.path.basename(inputfile))[0], shapesCount, rmse(result, np.asarray(img, dtype=np.float32)))
        cv2.imwrite(outputfile, result)

        plt.axis('off')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])