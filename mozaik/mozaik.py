#!/usr/bin/python

import sys, getopt
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import cv2
import tempfile, shutil
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
    gif = 0
    seed = None
    man = 'mozaik.py -i <inputfile> [-o <outputfile> -c <shapeCounts> -p <primitive> -m <method> -g <gif> -s <seed>]'
    try:
        opts, args = getopt.getopt(argv,"hi:o:c:p:m:g:s:",["input=","output=","count=","primitive=","method=","gif=","seed="])
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
        elif opt in ("-g", "--gif"):
            gif = int(arg)
        elif opt in ("-s", "--seed"):
            seed = arg
    if not inputfile:
        print "Input file needed"
        print man
        sys.exit(2)
    return (inputfile, outputfile, shapesCount, primitive, method, gif, seed)

def main(argv):
    (inputfile, outputfile, shapesCount, primitive, method, gif, seed) = parseArguments(argv)
    # Load image
    img = loadImage(inputfile)
    if img is None:
        print "Input file is not an image"
        sys.exit(2)

    # Process image
    random.seed(seed)
    [rendering, shapes] = method(shapesCount, primitive, maxSize=240, randomIterations=500).process(img)

    # Export rendering
    exportFilename = '{}_{}_{:1.0f}'.format(os.path.splitext(os.path.basename(inputfile))[0], shapesCount, Method.rmse(rendering, np.asarray(img, dtype=np.float32)))
    if gif:
        renderings = Method.renderIntermediates(shapes, Method.backgroundImage(img), gif)
        basesha1 = '%032x' % random.getrandbits(128)
        dirpath = tempfile.mkdtemp()
        for (i,render) in enumerate(renderings):
            cv2.imwrite('{}_{:03}.png'.format(dirpath+basesha1, i), render)
        os.system('convert -delay 75 {}_*.png {}.gif'.format(dirpath+basesha1, exportFilename))
        shutil.rmtree(dirpath)
    if not outputfile:
        outputfile = '{}.png'.format(exportFilename)
    cv2.imwrite(outputfile, rendering)

    # Show rendering
    plt.axis('off')
    plt.imshow(cv2.cvtColor(rendering, cv2.COLOR_BGR2RGB))
    plt.show()



if __name__ == "__main__":
   main(sys.argv[1:])