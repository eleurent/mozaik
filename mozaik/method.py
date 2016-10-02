from primitive import *

class Method(object):
    def __init__(self, shapesCount, primitive, maxSize):
        self.shapesCount = shapesCount
        self.primitive = primitive
        self.maxSize = maxSize

    def generateNewShape(self, canvas, reference):
        raise NotImplementedError()

    def process(self, img):
        (reference, canvas, result) = self.newCanvas(img)
        for i in range(self.shapesCount):
            shape = self.generateNewShape(canvas, reference)
            (canvas, result) = Method.addShape(shape, canvas, result)
            print '({}/{}) Canvas error: {:1.0f}'.format(i+1, self.shapesCount, Method.rmse(canvas, reference))
        return result

    def newCanvas(self, img):
        reference = np.asarray(Method.boundImage(img, self.maxSize), dtype=np.float32)
        canvas = Method.backgroundImage(reference)
        result = Method.backgroundImage(img)
        return (reference, canvas, result)

    @staticmethod
    def addShape(shape, canvas, result):
        return (shape.apply(canvas), shape.apply(result))

    @staticmethod
    def boundImage(img, maxSize):
        if img.shape[1] > maxSize:
            img = cv2.resize(img, (0,0), fx=float(maxSize)/img.shape[1], fy=float(maxSize)/img.shape[1])
        if img.shape[0] > maxSize:
            img = cv2.resize(img, (0,0), fx=float(maxSize)/img.shape[0], fy=float(maxSize)/img.shape[0])
        return img

    @staticmethod
    def backgroundImage(img):
        bgImg = np.zeros(img.shape, np.uint8)
        bgColor =  np.uint8(np.average(np.average(img, axis=0), axis=0))
        bgImg[:,:] = bgColor
        return bgImg

    @staticmethod
    def rmse(image, reference):
        return np.linalg.norm(reference - np.asarray(image, dtype=np.float32))

    @staticmethod
    def populationSelection(shapes, energies, n):
        indexes = energies.argpartition(n)[:n+1]
        return (shapes[indexes], energies[indexes])


class RandomMethod(Method):
    def __init__(self, shapesCount, primitive, maxSize, randomIterations):
        super(RandomMethod, self).__init__(shapesCount, primitive, maxSize)
        self.randomIterations = randomIterations

    def generateNewShape(self, canvas, reference):
        (shapes, errors) = self.generateRandomShapes(canvas, reference)
        return shapes[errors.argmin()]

    def generateRandomShapes(self, canvas, reference):
        shapes = np.array([self.primitive.generateRandom() for k in range(self.randomIterations)])
        errors = np.array([Method.rmse(shape.apply(canvas), reference) for shape in shapes])
        return (shapes, errors)


class HillClimbMethod(RandomMethod):
    def __init__(self, shapesCount, primitive, maxSize, randomIterations, selectionRate=0.25, hillClimbIterations=20):
        super(HillClimbMethod, self).__init__(shapesCount, primitive, maxSize, randomIterations)
        self.selectionRate = selectionRate
        self.hillClimbIterations = hillClimbIterations

    def generateNewShape(self, canvas, reference):
        [shapes, energies] = self.generateRandomShapes(canvas, reference)
        [shapes, energies] = Method.populationSelection(shapes, energies, n = min(int(self.randomIterations*self.selectionRate), self.randomIterations-1))
        # Hillclimbing of selected random shapes
        for j, shape in enumerate(shapes):
            energy = energies[j]
            for k in range(self.hillClimbIterations):
                newShape = shape.generateNeighbour()
                newEnergy = Method.rmse(newShape.apply(canvas), reference)
                if newEnergy < energy:
                    (shape, energy) = (newShape, newEnergy)
            (shapes[j], energies[j]) = (shape, energy)
        return shapes[energies.argmin()]


class SimulatedAnnealingMethod(RandomMethod):
    def __init__(self, shapesCount, primitive, maxSize, randomIterations, T0=30, Tf=0.01, tau=0.99):
        super(SimulatedAnnealingMethod, self).__init__(shapesCount, primitive, maxSize, randomIterations)
        self.T0 = T0
        self.Tf = Tf
        self.tau = tau

    def generateNewShape(self, canvas, reference):
        [shapes, energies] = self.generateRandomShapes(canvas, reference)
        shape = shapes[energies.argmin()]
        energy = Method.rmse(shape.apply(canvas), reference)
        T = self.T0
        while T > self.Tf:
            newShape = shape.generateNeighbour()
            newEnergy = Method.rmse(newShape.apply(canvas), reference)
            if newEnergy < energy or random.random() < np.exp(-(newEnergy-energy)/T):
                (shape, energy) = (newShape, newEnergy)
            T = self.tau*T
        return shape


class GradientDescentMethod(RandomMethod):
    def __init__(self, shapesCount, primitive, maxSize, randomIterations, selectionRate=0.25, gammaInit=1.0e-5, gammaMin=1.0e-7, gammaRate=0.3, h=0.01):
        super(GradientDescentMethod, self).__init__(shapesCount, primitive, maxSize, randomIterations)
        self.selectionRate = selectionRate
        self.gammaInit = gammaInit
        self.gammaMin = gammaMin
        self.gammaRate = gammaRate
        self.h = h

    def generateNewShape(self, canvas, reference):
        [shapes, energies] = self.generateRandomShapes(canvas, reference)
        [shapes, energies] = Method.populationSelection(shapes, energies, n = min(int(self.randomIterations*self.selectionRate), self.randomIterations-1))
        # Gradient descent of selected random shapes
        for j, shape in enumerate(shapes):
            energy = energies[j]
            gamma = self.gammaInit
            while gamma > self.gammaMin:
                g = GradientDescentMethod.computeGradient(shape, self.h, canvas, reference)
                newShape = shape.generateFromState(shape.getState() - g*gamma)
                newEnergy = Method.rmse(newShape.apply(canvas), reference)
                if newEnergy < energy:
                    (shape, energy) = (newShape, newEnergy)
                else:
                    gamma *= self.gammaRate
            (shapes[j], energies[j]) = (shape, energy)
        return shapes[energies.argmin()]

    @staticmethod
    def computeGradient(shape, h, canvas, reference):
        energy = Method.rmse(shape.apply(canvas), reference)
        X = shape.getState()
        I = np.identity(X.size)
        Xh = X+I*h
        g = (np.array([Method.rmse(shape.generateFromState(Xh[i,:]).apply(canvas), reference) for i in range(X.size)]) - energy)/h
        return g
