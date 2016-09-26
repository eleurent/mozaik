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
        size = img.shape[1::-1]
        cv2.rectangle(overlay, scale(self.positionA, size), scale(self.positionB, size), scale(self.color, 255), -1)
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    def generateNeighbour(self):
        WINDOW = 0.1
        def eps():
            return WINDOW*(2*random.random()-1)
        size = np.absolute(np.array(self.positionB)-np.array(self.positionA))
        center = (np.array(self.positionA)+np.array(self.positionB))/2
        size = np.clip(size*np.array([1+eps(), 1+eps()]), 0.05, 1)
        center += np.array([eps(), eps()])
        positionA = np.clip(center-size/2, 0, 1)
        positionB = np.clip(center+size/2, 0, 1)
        color = np.clip(np.asarray(self.color)+np.array([eps(), eps(), eps()]), 0, 1)
        alpha = np.clip(self.alpha+eps(), 0, 1)
        return RectanglePrimitive(positionA, positionB, color, alpha)

    @staticmethod
    def generateRandom():
        positionA = (random.random(), random.random())
        positionB = (random.random(), random.random())
        color = (random.random(), random.random(), random.random())
        alpha = random.random()
        return RectanglePrimitive(positionA, positionB, color, alpha)

    def __str__(self):
        return '{} {} - {} x {}'.format(self.positionA, self.positionB, self.color, self.alpha)

def scale(data, ratio):
    return tuple((np.asarray(data)*np.asarray(ratio)).astype(int))

def rmse(image, reference):
    return np.linalg.norm(reference - np.asarray(image, dtype=np.float32))

def loadImage(filename):
    return cv2.imread(filename)

def randomGeneration(img, shapesCount, iterations):
    reference = np.asarray(img, dtype=np.float32)
    canvas = np.zeros(img.shape, np.uint8)
    for i in range(shapesCount):
        shapes = [RectanglePrimitive.generateRandom() for k in range(iterations)]
        results = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        bestShape = shapes[results.argmin()]
        canvas = bestShape.apply(canvas)
        print '({}/{}) Canvas error: {}'.format(i+1, shapesCount, rmse(canvas, reference))
    return canvas

def annealingGeneration(img, shapesCount, randomIterations, T0, Tf, tau):
    reference = np.asarray(img, dtype=np.float32)
    canvas = np.zeros(img.shape, np.uint8)
    for i in range(shapesCount):
        shapes = [RectanglePrimitive.generateRandom() for k in range(randomIterations)]
        results = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        shape = shapes[results.argmin()]
        energy = rmse(shape.apply(canvas), reference)
        # print "Selected random seed: {}".format(energy)
        T = T0
        while T > Tf:
            newShape = shape.generateNeighbour()
            newEnergy = rmse(newShape.apply(canvas), reference)
            if newEnergy < energy or random.random() < np.exp(-(newEnergy-energy)/T):
                shape = newShape
                energy = newEnergy
            T = tau*T
        canvas = shape.apply(canvas)
        print '({}/{}) Canvas error: {:1.0f}'.format(i+1, shapesCount, rmse(canvas, reference))
    return canvas

def main(argv):
    inputfile = ''
    outputfile = ''
    shapesCount = 200
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
        elif opts in ("-c", "--count"):
            shapesCount = arg
    if not inputfile:
        print 'mozaik.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    else:
        random.seed()
        img = loadImage(inputfile)
        # result = randomGeneration(img, shapesCount, iterations=1000)
        result = annealingGeneration(img, shapesCount, randomIterations=40, T0=50, Tf=1, tau=0.97)
        # result = annealingGeneration(img, shapesCount, randomIterations=400, T0=50, Tf=0.1, tau=0.99)

        if not outputfile:
            outputfile = 'out_{}_{:1.0f}.png'.format(shapesCount, rmse(result, np.asarray(img, dtype=np.float32)))
        cv2.imwrite(outputfile, result)

        plt.axis('off')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])