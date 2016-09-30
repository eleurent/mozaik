from primitive import *

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
