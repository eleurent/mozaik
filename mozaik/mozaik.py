#!/usr/bin/python

import sys, getopt
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import cv2

from primitive import RectanglePrimitive, TrianglePrimitive, EllipsePrimitive, CirclePrimitive, SquarePrimitive
from method import Method, RandomMethod, HillClimbMethod, SimulatedAnnealingMethod, GradientDescentMethod

def loadImage(filename):
    return cv2.imread(filename)

def parseArguments(argv):
    inputfile = ''
    outputfile = ''
    shapesCount = 200
    primitive = RectanglePrimitive
    method = GradientDescentMethod
    seed = None
    man = 'mozaik.py -i <inputfile> [-o <outputfile> -c <shapeCounts> -p <primitive> -m <method> -s <seed>]'
    try:
        opts, args = getopt.getopt(argv,"hi:o:c:p:m:s:",["input=","output=","count=","primitive="])
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
                print 'Primitive not supported'
                sys.exit(2)
        elif opt in ("-m", "--method"):
            if arg == "random":
                method = RandomMethod
            elif arg == "hillclimb":
                method = HillClimbMethod
            elif arg == "annealing":
                method = SimulatedAnnealingMethod
            elif arg == "gradient":
                method = GradientDescentMethod
            else:
                print 'Method not supported'
                sys.exit(2)
        elif opt in ("-s", "--seed"):
            seed = arg
    if not inputfile:
        print "Input file needed"
        print man
        sys.exit(2)
    return (inputfile, outputfile, shapesCount, primitive, method, seed)

def main(argv):
    (inputfile, outputfile, shapesCount, primitive, method, seed) = parseArguments(argv)
    # Load image
    img = loadImage(inputfile)
    if img is None:
        print "Input file is not an image"
        sys.exit(2)
    # Process image
    random.seed(seed)
    result = method(shapesCount, primitive, maxSize=240, randomIterations=100).process(img)
    # Export result
    if not outputfile:
        outputfile = '{}_{}_{:1.0f}.png'.format(os.path.splitext(os.path.basename(inputfile))[0], shapesCount, Method.rmse(result, np.asarray(img, dtype=np.float32)))
    cv2.imwrite(outputfile, result)
    # Show result
    plt.axis('off')
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])