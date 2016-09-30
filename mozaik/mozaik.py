#!/usr/bin/python

import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os
import math

class Primitive(object):
    def apply(self, img):
        raise NotImplementedError()

    def generateNeighbour(self):
        raise NotImplementedError()

    @classmethod
    def generateRandom(cls):
        raise NotImplementedError()

    @classmethod
    def generateFromState(cls, X):
        raise NotImplementedError()

    def getState(self):
        raise NotImplementedError()

    def computeGradient(self, h, canvas, reference):
        energy = rmse(self.apply(canvas), reference)
        X = self.getState()
        I = np.identity(X.size)
        Xh = X+I*h
        g = (np.array([rmse(self.__class__.generateFromState(Xh[i,:]).apply(canvas), reference) for i in range(X.size)]) - energy)/h
        return g

    def __str__(self):
        raise NotImplementedError()

class RectanglePrimitive(Primitive):
    def __init__(self, positionA, positionB, color, alpha):
        self.positionA = np.clip(positionA, 0, 1)
        self.positionB = np.clip(positionB, 0, 1)
        self.color = np.clip(color, 0, 1)
        self.alpha = np.clip(alpha, 0, 1)

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
        positionA = center-size/2
        positionB = center+size/2
        color = self.color+np.array([eps(), eps(), eps()])
        alpha = self.alpha+eps()
        return RectanglePrimitive(positionA, positionB, color, alpha)

    @classmethod
    def generateRandom(cls):
        positionA = np.array([random.random(), random.random()])
        positionB = np.array([random.random(), random.random()])
        color = np.array([random.random(), random.random(), random.random()])
        alpha = random.random()
        return RectanglePrimitive(positionA, positionB, color, alpha)

    @classmethod
    def generateFromState(cls, X):
        if X.size != 8:
            return None
        return RectanglePrimitive(np.array([X[0],X[1]]),
            np.array([X[2],X[3]]),
            np.array([X[4],X[5],X[6]]),
            X[7])

    def getState(self):
        return np.hstack((self.positionA, self.positionB, self.color, self.alpha))


    def __str__(self):
        return '{} {} - {} x {}'.format(self.positionA, self.positionB, self.color, self.alpha)

class TrianglePrimitive(Primitive):
    def __init__(self, positionA, positionB, positionC, color, alpha):
        self.positionA = np.clip(positionA, 0, 1)
        self.positionB = np.clip(positionB, 0, 1)
        self.positionC = np.clip(positionC, 0, 1)
        self.color = np.clip(color, 0, 1)
        self.alpha = np.clip(alpha, 0, 1)

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
        positionA = center+rel[0]
        positionB = center+rel[1]
        positionC = center+rel[2]
        color = np.asarray(self.color)+np.array([eps(), eps(), eps()])
        alpha = self.alpha+eps()
        return TrianglePrimitive(positionA, positionB, positionC, color, alpha)

    @classmethod
    def generateRandom(cls):
        positionA = np.array([random.random(), random.random()])
        positionB = np.array([random.random(), random.random()])
        positionC = np.array([random.random(), random.random()])
        color = np.array([random.random(), random.random(), random.random()])
        alpha = random.random()
        return TrianglePrimitive(positionA, positionB, positionC, color, alpha)

    @classmethod
    def generateFromState(cls, X):
        if X.size != 10:
            return None
        return TrianglePrimitive(np.array([X[0],X[1]]),
            np.array([X[2],X[3]]),
            np.array([X[4],X[5]]),
            np.array([X[6],X[7],X[8]]),
            X[9])

    def getState(self):
        return np.hstack((self.positionA, self.positionB, self.positionC, self.color, self.alpha))

    def __str__(self):
        return '{} {} {} - {} x {}'.format(self.positionA, self.positionB, self.positionC, self.color, self.alpha)


class EllipsePrimitive(Primitive):
    def __init__(self, center, axes, angle, color, alpha):
        self.center = center
        self.axes = np.clip(axes, 0.01, 1)
        self.angle = np.clip(angle, 0, 1)
        self.color = np.clip(color, 0, 1)
        self.alpha = np.clip(alpha, 0, 1)

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
        angle = self.alpha+eps()
        color = self.color+np.array([eps(), eps(), eps()])
        alpha = self.alpha+eps()
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
        ellipse = super(CirclePrimitive, self).generateNeighbour()
        return CirclePrimitive(ellipse.center, ellipse.axes, ellipse.angle, ellipse.color, ellipse.alpha)

    @classmethod
    def generateRandom(cls):
        ellipse = super(CirclePrimitive, cls).generateRandom()
        return CirclePrimitive(ellipse.center, ellipse.axes, ellipse.angle, ellipse.color, ellipse.alpha)

    def __str__(self):
        return super(CirclePrimitive, self).__str__()


class SquarePrimitive(RectanglePrimitive):
    def __init__(self, positionA, positionB, color, alpha):
        super(SquarePrimitive, self).__init__(positionA, positionB, color, alpha)

    def apply(self, img):
        overlay = img.copy()
        output = img.copy()
        size = img.shape[1::-1]
        center = (self.positionA+self.positionB)/2*size
        side = math.sqrt(abs(np.prod(self.positionB-self.positionA)*np.prod(size)))
        cv2.rectangle(overlay,scale(center+side/2,1),scale(center-side/2,1),scale(self.color, 255),-1)
        cv2.addWeighted(overlay, self.alpha, output, 1 - self.alpha, 0, output)
        return output

    def generateNeighbour(self):
        rectangle = super(SquarePrimitive, self).generateNeighbour()
        return SquarePrimitive(rectangle.positionA, rectangle.positionB, rectangle.color, rectangle.alpha)

    @classmethod
    def generateRandom(cls):
        rectangle = super(SquarePrimitive, cls).generateRandom()
        return SquarePrimitive(rectangle.positionA, rectangle.positionB, rectangle.color, rectangle.alpha)

    def __str__(self):
        return super(SquarePrimitive, self).__str__()



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

def initBackground(img):
    bgImg = np.zeros(img.shape, np.uint8)
    bgColor =  np.uint8(np.average(np.average(img, axis=0), axis=0))
    bgImg[:,:] = bgColor
    return bgImg

def randomGeneration(img, shapesCount, primitive, maxSize, randomIterations):
    reference = np.asarray(resize(img, maxSize), dtype=np.float32)
    canvas = initBackground(reference)
    result = initBackground(img)
    for i in range(shapesCount):
        shapes = [primitive.generateRandom() for k in range(randomIterations)]
        energies = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        bestShape = shapes[energies.argmin()]
        canvas = bestShape.apply(canvas)
        result = bestShape.apply(result)
        print '({}/{}) Canvas error: {:1.0f}'.format(i+1, shapesCount, rmse(canvas, reference))
    return result

def hillClimbGeneration(img, shapesCount, primitive, maxSize, randomIterations, randomPopSelection, hillClimbIterations):
    reference = np.asarray(resize(img, maxSize), dtype=np.float32)
    canvas = initBackground(reference)
    result = initBackground(img)
    for i in range(shapesCount):
        # Random population generation
        shapes = [primitive.generateRandom() for k in range(randomIterations)]
        energies = [rmse(shape.apply(canvas), reference) for shape in shapes]
        # Random population selection
        n = min(int(randomIterations*randomPopSelection), randomPopSelection-1)
        indexes = energies.argpartition(n)[:n+1]
        shapes = shapes[indexes]
        energies = energies[indexes]
        for j, shape in enumerate(shapes):
            energy = energies[j]
            # Hillclimbing of selected random shapes
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
        print '({}/{}) Canvas error: {:1.0f}'.format(i+1, shapesCount, rmse(canvas, reference))
    return result

def annealingGeneration(img, shapesCount, primitive, maxSize, randomIterations, T0, Tf, tau):
    reference = np.asarray(resize(img, maxSize), dtype=np.float32)
    canvas = initBackground(reference)
    result = initBackground(img)
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

def gradientGeneration(img, shapesCount, primitive, maxSize, randomIterations, randomPopSelection):
    reference = np.asarray(resize(img, maxSize), dtype=np.float32)
    canvas = initBackground(reference)
    result = initBackground(img)
    h = 0.01
    for i in range(shapesCount):
        # Random population generation
        shapes = np.array([primitive.generateRandom() for k in range(randomIterations)])
        energies = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        # Random population selection
        n = min(int(randomIterations*randomPopSelection), randomIterations-1)
        indexes = np.argpartition(energies, n)[:n+1]
        shapes = shapes[indexes]
        energies = energies[indexes]
        for j, shape in enumerate(shapes):
            energy = energies[j]
            # Gradient descent of selected random shapes
            beta = 1.0e-6
            while beta > 1.0e-9:
                g = shape.computeGradient(h, canvas, reference)
                newShape = primitive.generateFromState(shape.getState() - g*beta)
                newEnergy = rmse(newShape.apply(canvas), reference)
                if newEnergy < energy:
                    shape = newShape
                    energy = newEnergy
                else:
                    beta = beta*0.3
            shapes[j] = shape
        energies = np.array([rmse(shape.apply(canvas), reference) for shape in shapes])
        bestShape = shapes[energies.argmin()]
        canvas = bestShape.apply(canvas)
        result = bestShape.apply(result)
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