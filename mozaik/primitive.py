import numpy as np
import cv2
import random
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