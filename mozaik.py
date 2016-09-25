#!/usr/bin/python

import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

class RectanglePrimitive:
    def __init__(self, positionA, positionB, color, alpha):
        self.positionA = positionA
        self.positionB = positionB
        self.color = color
        self.alpha = alpha

    def apply(self, img):
        overlay = img.copy()
        output = img.copy()
        cv2.rectangle(overlay, self.positionA, self.positionB, self.color, -1)
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    @staticmethod
    def generateRandom(width, height):
        positionA = (random.randint(0,width-1), random.randint(0,height-1))
        positionB = (random.randint(0,width-1), random.randint(0,height-1))
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        alpha = random.random()
        return RectanglePrimitive(positionA, positionB, color, alpha)

    def __str__(self):
        return '{} {} - {} x {}'.format(self.positionA, self.positionB, self.color, self.alpha)

def energy(image, reference):
    return np.linalg.norm(reference - np.asarray(image, dtype=np.float32))

def loadImage(filename):
    return cv2.imread(filename)

def randomGeneration(img, shapesCount, iterations):
    reference = np.asarray(img, dtype=np.float32)
    height, width = img.shape[:2]
    canvas = np.zeros((height,width,3), np.uint8)
    for i in range(shapesCount):
        shapes = [RectanglePrimitive.generateRandom(width, height) for k in range(iterations)]
        results = np.array([energy(shape.apply(canvas), reference) for shape in shapes])
        bestShape = shapes[results.argmin()]
        canvas = bestShape.apply(canvas)
        print '({}/{}) Canvas error: {}'.format(i+1, iterations, energy(canvas, reference))
    return canvas


def main(argv):
    inputfile = ''
    outputfile = 'output.png'
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["input=","output="])
    except getopt.GetoptError:
        print 'mozaik.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'mozaik.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-o", "--output"):
            outputfile = arg
    if not inputfile:
        print 'mozaik.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    else:
        result = randomGeneration(loadImage(inputfile), 50, 100)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(outputfile, result)


if __name__ == "__main__":
   main(sys.argv[1:])