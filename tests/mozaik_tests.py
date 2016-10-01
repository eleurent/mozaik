from nose.tools import *
import mozaik.primitive
from mozaik.primitive import RectanglePrimitive, TrianglePrimitive, EllipsePrimitive, CirclePrimitive, SquarePrimitive
import numpy as np

class TestPrimitive:
    img = []

    def setup(self):
        pass

    def teardown(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.img = np.zeros((5,5,3))

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