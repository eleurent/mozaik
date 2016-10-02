from nose.tools import *
from mozaik.primitive import RectanglePrimitive, TrianglePrimitive, EllipsePrimitive, CirclePrimitive, SquarePrimitive
from mozaik.method import Method, RandomMethod, HillClimbMethod, SimulatedAnnealingMethod, GradientDescentMethod
import numpy as np
import cv2

class TestPrimitive:
    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.img = np.zeros((100,100,3))

    @classmethod
    def teardown_class(cls):
        pass

    def test_rectangle_full(self):
        color = np.array([10,20,30])
        rect = RectanglePrimitive((0,0),(1,1),color/255.,1)
        canvas = rect.apply(TestPrimitive.img)
        assert canvas.size == TestPrimitive.img.size
        assert np.all(canvas == color)

    def assert_random_primitive(self, primitive):
        shape = primitive.generateRandom()
        canvas = shape.apply(TestPrimitive.img)
        assert canvas.size == TestPrimitive.img.size
        assert np.any(canvas) or not np.any(shape.color) or not shape.alpha

    def test_random_rectangle(self):
        self.assert_random_primitive(RectanglePrimitive)

    def test_random_triangle(self):
        self.assert_random_primitive(TrianglePrimitive)

    def test_random_ellipse(self):
        self.assert_random_primitive(EllipsePrimitive)

    def test_random_circle(self):
        self.assert_random_primitive(CirclePrimitive)

    def test_random_square(self):
        self.assert_random_primitive(SquarePrimitive)

    def assert_neighbour_primitive(self, primitive):
        shapeA = primitive.generateRandom()
        shapeB = shapeA.generateNeighbour()
        threshold = 0.1
        assert shapeA.__class__ == shapeB.__class__
        assert abs(shapeA.alpha - shapeB.alpha) < threshold
        assert np.all(abs(shapeA.color - shapeB.color) < threshold)
        if hasattr(primitive, 'positionA'):
            assert np.all(abs(shapeA.positionA - shapeB.positionA) < threshold)
        if hasattr(primitive, 'positionB'):
            assert np.all(abs(shapeA.positionB - shapeB.positionB) < threshold)
        if hasattr(primitive, 'positionC'):
            assert np.all(abs(shapeA.positionC - shapeB.positionC) < threshold)
        if hasattr(primitive, 'center'):
            assert np.all(abs(shapeA.center - shapeB.center) < threshold)
        if hasattr(primitive, 'axes'):
            assert np.all(abs(shapeA.axes - shapeB.axes) < threshold)

    def test_neighbour_rectangle(self):
        self.assert_neighbour_primitive(RectanglePrimitive)

    def test_neighbour_triangle(self):
        self.assert_neighbour_primitive(TrianglePrimitive)

    def test_neighbour_ellipse(self):
        self.assert_neighbour_primitive(EllipsePrimitive)

    def test_neighbour_circle(self):
        self.assert_neighbour_primitive(CirclePrimitive)

    def test_neighbour_square(self):
        self.assert_neighbour_primitive(SquarePrimitive)

class TestMethod:
    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.img = np.zeros((100,100,3), dtype='uint8')
        cv2.rectangle(cls.img, (0,0), (50,50), (255,0,0), -1)
        cv2.rectangle(cls.img, (50,0), (100,50), (0,255,0), -1)
        cv2.rectangle(cls.img, (0,50), (50,100), (0,0,255), -1)
        cv2.rectangle(cls.img, (50,50), (100,100), (255,255,0), -1)

    @classmethod
    def teardown_class(cls):
        pass

    def assert_method(self, method, threshold):
        print method
        result = method.process(self.img)
        error = Method.rmse(result,  np.asarray(self.img, dtype = np.float32))
        assert result.size == self.img.size
        assert error < threshold

    def test_random_method(self):
        self.assert_method(RandomMethod(shapesCount=4, primitive=RectanglePrimitive, maxSize=240, randomIterations=1000), 19000)

    def test_hillclimb_method(self):
        self.assert_method(HillClimbMethod(shapesCount=4, primitive=RectanglePrimitive, maxSize=240, randomIterations=100), 17000)

    def test_annealing_method(self):
        self.assert_method(SimulatedAnnealingMethod(shapesCount=4, primitive=RectanglePrimitive, maxSize=240, randomIterations=100), 12000)

    def test_gradient_descent_method(self):
        self.assert_method(GradientDescentMethod(shapesCount=4, primitive=RectanglePrimitive, maxSize=240, randomIterations=100), 17000)
