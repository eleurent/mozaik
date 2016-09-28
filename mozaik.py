#!/usr/bin/python

import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os

class Primitive(object):
    def apply(self, img):
        raise NotImplementedError()

    def generateNeighbour(self):
        raise NotImplementedError()

    @classmethod
    def generateRandom(cls):
        # raise NotImplementedError()
        pass

    def __str__(self):
        raise NotImplementedError()

class RectanglePrimitive(Primitive):
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
        size = np.absolute(self.positionB-self.positionA)
        center = (self.positionA+self.positionB)/2
        size = np.clip(size*np.array([1+eps(), 1+eps()]), 0.05, 1)
        center += np.array([eps(), eps()])
        positionA = np.clip(center-size/2, 0, 1)
        positionB = np.clip(center+size/2, 0, 1)
        color = np.clip(self.color+np.array([eps(), eps(), eps()]), 0, 1)
        alpha = np.clip(self.alpha+eps(), 0, 1)
        return RectanglePrimitive(positionA, positionB, color, alpha)

    @classmethod
    def generateRandom(cls):
        positionA = np.array([random.random(), random.random()])
        positionB = np.array([random.random(), random.random()])
        color = np.array([random.random(), random.random(), random.random()])
        alpha = random.random()
        return RectanglePrimitive(positionA, positionB, color, alpha)

    def __str__(self):
        return '{} {} - {} x {}'.format(self.positionA, self.positionB, self.color, self.alpha)

class TrianglePrimitive(Primitive):
    def __init__(self, positionA, positionB, positionC, color, alpha):
        self.positionA = positionA
        self.positionB = positionB
        self.positionC = positionC
        self.color = color
        self.alpha = alpha

    def apply(self, img):
        overlay = img.copy()
        output = img.copy()
        size = img.shape[1::-1]
        pts = np.array([scale(self.positionA, size), scale(self.positionB, size), scale(self.positionC, size)], np.int32).reshape((-1,1,2))
        cv2.fillConvexPoly(overlay,pts,scale(self.color, 255))
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    def generateNeighbour(self):
        WINDOW = 0.1
        def eps():
            return WINDOW*(2*random.random()-1)
        center = (self.positionA+self.positionB+self.positionC)/3
        rel = [self.positionA-center, self.positionB-center, self.positionC-center]
        center += np.array([eps(), eps()])
        rel *= np.array([1+eps(), 1+eps()])
        positionA = np.clip(center+rel[0], 0, 1)
        positionB = np.clip(center+rel[1], 0, 1)
        positionC = np.clip(center+rel[2], 0, 1)
        color = np.clip(np.asarray(self.color)+np.array([eps(), eps(), eps()]), 0, 1)
        alpha = np.clip(self.alpha+eps(), 0, 1)
        return TrianglePrimitive(positionA, positionB, positionC, color, alpha)

    @classmethod
    def generateRandom(cls):
        positionA = np.array([random.random(), random.random()])
        positionB = np.array([random.random(), random.random()])
        positionC = np.array([random.random(), random.random()])
        color = np.array([random.random(), random.random(), random.random()])
        alpha = random.random()
        return TrianglePrimitive(positionA, positionB, positionC, color, alpha)

    def __str__(self):
        return '{} {} {} - {} x {}'.format(self.positionA, self.positionB, self.positionC, self.color, self.alpha)


class EllipsePrimitive(Primitive):
    def __init__(self, center, axes, angle, color, alpha):
        self.center = center
        self.axes = axes
        self.angle = angle
        self.color = color
        self.alpha = alpha

    def apply(self, img):
        overlay = img.copy()
        output = img.copy()
        size = img.shape[1::-1]
        cv2.ellipse(overlay,scale(self.center, size),scale(self.axes, size),self.angle*360,0,360,scale(self.color, 255),-1)
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    def generateNeighbour(self):
        WINDOW = 0.1
        def eps():
            return WINDOW*(2*random.random()-1)
        center = self.center + np.array([eps(), eps()])
        axes = self.axes * np.array([1+eps(), 1+eps()])
        angle = np.clip(self.alpha+eps(), 0, 1)
        color = np.clip(np.asarray(self.color)+np.array([eps(), eps(), eps()]), 0, 1)
        alpha = np.clip(self.alpha+eps(), 0, 1)
        return EllipsePrimitive(center, axes, angle, color, alpha)

    @classmethod
    def generateRandom(cls):
        center = np.array([random.random(), random.random()])
        axes = np.array([random.random()/2, random.random()/2])
        angle = random.random()
        color = np.array([random.random(), random.random(), random.random()])
        alpha = random.random()
        return EllipsePrimitive(center, axes, angle, color, alpha)

    def __str__(self):
        return '{} {} {} - {} x {}'.format(self.center, self.axes, self.angle, self.color, self.alpha)


class CirclePrimitive(EllipsePrimitive):
    def __init__(self, center, axes, angle, color, alpha):
        super(CirclePrimitive, self).__init__(center, np.array(axes[0], axes[0]), angle, color, alpha)

    def apply(self, img):
        overlay = img.copy()
        output = img.copy()
        size = img.shape[1::-1]
        cv2.ellipse(overlay,scale(self.center, size),scale(self.axes, size[0]),self.angle*360,0,360,scale(self.color, 255),-1)
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    def generateNeighbour(self):
        neighbour = super(CirclePrimitive, self).generateNeighbour()
        neighbour.__class__ = CirclePrimitive
        neighbour.axes[1] = neighbour.axes[0]
        return neighbour

    @classmethod
    def generateRandom(cls):
        neighbour = super(CirclePrimitive, cls).generateRandom()
        neighbour.__class__ = CirclePrimitive
        neighbour.axes[1] = neighbour.axes[0]
        return neighbour

    def __str__(self):
        return super(CirclePrimitive, self).__str__()


def scale(data, ratio):
    return tuple((np.asarray(data)*np.asarray(ratio)).astype(int))

def rmse(image, reference):
    return np.linalg.norm(reference - np.asarray(image, dtype=np.float32))

def loadImage(filename):
    return cv2.imread(filename)

def resize(img, maxSize):
    if img.shape[1] > maxSize:
        img = cv2.resize(img, (0,0), fx=float(maxSize)/img.shape[1], fy=float(maxSize)/img.shape[1])
    if img.shape[0] > maxSize:
        img = cv2.resize(img, (0,0), fx=float(maxSize)/img.shape[0], fy=float(maxSize)/img.shape[0])
    return img

def randomGeneration(img, shapesCount, primitive, maxSize, randomIterations):
    reference = np.asarray(resize(img, maxSize), dtype=np.float32)
    canvas = np.zeros(reference.shape, np.uint8)
    result = np.zeros(img.shape, np.uint8)
    for i in range(shapesCount):
        shapes = [primitive.generateRandom() for k in range(randomIterations)]
        energies = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        bestShape = shapes[energies.argmin()]
        canvas = bestShape.apply(canvas)
        result = bestShape.apply(result)
        print '({}/{}) Canvas error: {}'.format(i+1, shapesCount, rmse(canvas, reference))
    return result

def hillClimbGeneration(img, shapesCount, primitive, maxSize, randomIterations, hillClimbIterations):
    reference = np.asarray(resize(img, maxSize), dtype=np.float32)
    canvas = np.zeros(reference.shape, np.uint8)
    result = np.zeros(img.shape, np.uint8)
    for i in range(shapesCount):
        shapes = [primitive.generateRandom() for k in range(randomIterations)]
        for j, shape in enumerate(shapes):
            energy = rmse(shape.apply(canvas), reference)
            for k in range(hillClimbIterations):
                newShape = shape.generateNeighbour()
                newEnergy = rmse(newShape.apply(canvas), reference)
                if newEnergy < energy:
                    shape = newShape
                    energy = newEnergy
                shapes[j] = shape
        energies = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        bestShape = shapes[energies.argmin()]
        canvas = bestShape.apply(canvas)
        result = bestShape.apply(result)
        print '({}/{}) Canvas error: {}'.format(i+1, shapesCount, rmse(canvas, reference))
    return result

def annealingGeneration(img, shapesCount, primitive, maxSize, randomIterations, T0, Tf, tau):
    reference = np.asarray(resize(img, maxSize), dtype=np.float32)
    canvas = np.zeros(reference.shape, np.uint8)
    result = np.zeros(img.shape, np.uint8)
    for i in range(shapesCount):
        shapes = [primitive.generateRandom() for k in range(randomIterations)]
        energies = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        k = energies.argmin()
        shape = shapes[k]
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
        result = shape.apply(result)
        print '({}/{}) Canvas error: {:1.0f}'.format(i+1, shapesCount, rmse(canvas, reference))
    return result

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
            else:
                print 'Primitive not recognized'
                sys.exit(2)
    if not inputfile:
        print man
        sys.exit(2)
    else:
        random.seed()
        img = loadImage(inputfile)
        # result = randomGeneration(img, shapesCount, primitive, maxSize=256, randomIterations=4000)
        # result = annealingGeneration(img, shapesCount, primitive, maxSize=256, randomIterations=40, T0=50, Tf=1, tau=0.97)
        # result = annealingGeneration(img, shapesCount, primitive, maxSize=256, randomIterations=400, T0=30, Tf=0.01, tau=0.99)
        result = hillClimbGeneration(img, shapesCount, primitive, maxSize=256, randomIterations=100, hillClimbIterations=10)


        if not outputfile:
            outputfile = '{}_{}_{:1.0f}.png'.format(os.path.splitext(os.path.basename(inputfile))[0], shapesCount, rmse(result, np.asarray(img, dtype=np.float32)))
        cv2.imwrite(outputfile, result)

        plt.axis('off')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])