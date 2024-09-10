import numpy as np
from numpy.testing import assert_equal, assert_array_less, assert_allclose
import exercices


class TestNiveau1:
    def test_create_zeros(self):
        actual = exercices.create_zeros()
        expected = [0, 0, 0, 0, 0]
        assert_equal(actual, expected)

    def test_create_ones(self):
        actual = exercices.create_ones()
        expected = [1, 1, 1, 1, 1]
        assert_equal(actual, expected)

    def test_create_range(self):
        actual = exercices.create_range()
        # fmt: off
        expected = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                    44, 45, 46, 47, 48, 49, 50]
        # fmt: on
        assert_equal(actual, expected)

    def test_create_identity(self):
        actual = exercices.create_identity()
        # fmt: off
        expected = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
        # fmt: on
        assert_equal(actual, expected)

    def test_create_random(self):
        actual = exercices.create_random()
        assert actual.shape == (3, 3)
        assert not np.all(actual == 0)
        assert not np.all(actual == 1)
        assert_array_less(actual, 1, strict=False)
        assert_array_less(0, actual, strict=False)
        # assert np.all((actual >= 0) & (actual <= 1))


class TestNiveau2:
    def test_add_five(self):
        # Cas simle
        actual = exercices.add_five(np.array([1, 2, 3, 4, 5]))
        expected = [6, 7, 8, 9, 10]
        assert_equal(actual, expected)

        # Cas aux limites (ne doit pas "crasher")
        actual = exercices.add_five(np.array([]))
        expected = []
        assert_equal(actual, expected)

    def test_square(self):
        # Cas simle
        actual = exercices.square(np.array([1, 2, 3, 4, 5]))
        expected = [1, 4, 9, 16, 25]
        assert_equal(actual, expected)

        # Cas aux limites (ne doit pas "crasher")
        actual = exercices.square(np.array([]))
        expected = []
        assert_equal(actual, expected)

    def test_sin_values(self):
        actual = exercices.sin_values()
        # fmt: off
        expected = [ 0.        ,  0.09983342,  0.19866933,  0.29552021,  0.38941834,
                     0.47942554,  0.56464247,  0.64421769,  0.71735609,  0.78332691,
                     0.84147098,  0.89120736,  0.93203909,  0.96355819,  0.98544973,
                     0.99749499,  0.9995736 ,  0.99166481,  0.97384763,  0.94630009,
                     0.90929743,  0.86320937,  0.8084964 ,  0.74570521,  0.67546318,
                     0.59847214,  0.51550137,  0.42737988,  0.33498815,  0.23924933,
                     0.14112001,  0.04158066, -0.05837414, -0.15774569, -0.2555411 ,
                    -0.35078323, -0.44252044, -0.52983614, -0.61185789, -0.68776616,
                    -0.7568025 , -0.81827711, -0.87157577, -0.91616594, -0.95160207,
                    -0.97753012, -0.993691  , -0.99992326, -0.99616461, -0.98245261,
                    -0.95892427, -0.92581468, -0.88345466, -0.83226744, -0.77276449,
                    -0.70554033, -0.63126664, -0.55068554, -0.46460218, -0.37387666,
                    -0.2794155 , -0.1821625 , -0.0830894 ,  0.0168139 ]
        # fmt: on
        assert_allclose(actual, expected)

    def test_f_vectorized(self):
        # on teste avec des valeurs aléatoires
        arr1 = np.random.randint(0, 10, 5)
        arr2 = np.random.randint(0, 10, 5)
        actual = exercices.f_vectorized(arr1, arr2)
        expected = exercices.f(arr1, arr2)
        assert_equal(actual, expected)

        # et aussi on vérifie que f n'a pas été modifiée
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([5, 4, 3, 2, 1])
        actual = exercices.f(arr1, arr2)
        expected = [17, 16, 15, 14, 13]
        assert_equal(actual, expected)

    def test_g_vectorized(self):
        # on teste avec des valeurs aléatoires
        arr = np.random.randint(-10, 10, 5)
        actual = exercices.g_vectorized(arr)
        expected = exercices.g(arr)
        assert_equal(actual, expected)

        # Au cas où...
        arr = np.array([1, -2, 3, -4, 5])
        actual = exercices.g_vectorized(arr)
        expected = [1, -2, 9, -4, 25]
        assert_equal(actual, expected)


class TestNiveau3:
    def test_select_event(self):
        # Cas simple
        test_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual = exercices.select_even(test_arr)
        expected = [2, 4, 6, 8, 10]
        assert_equal(actual, expected)

        # Des cas aux limites
        actual = []
        expected = []
        assert_equal(actual, expected)

        actual = exercices.select_even(np.array([1, 3, 5]))
        expected = []
        assert_equal(actual, expected)

    def test_replace_negatives(self):
        # l'exemple doit marcher
        test_arr = np.array([1, -2, 3, -4, 5])
        actual = exercices.replace_negatives(test_arr)
        expected = [1, 0, 3, 0, 5]
        assert_equal(actual, expected)

        # cas aux limites
        actual = []
        expected = []
        assert_equal(actual, expected)

        actual = exercices.replace_negatives(np.array([1, 2, 3]))
        expected = [1, 2, 3]
        assert_equal(actual, expected)

    def test_get_center(self):
        # l'exemple doit marcher
        test_arr = np.array(np.arange(1, 26).reshape(5, 5))
        actual = exercices.get_center(test_arr)
        expected = [[7, 8, 9], [12, 13, 14], [17, 18, 19]]
        assert_equal(actual, expected)

        # cas aux limites
        test_arr = np.array([[1, 2], [3, 4]])
        actual = exercices.get_center(test_arr)
        expected = np.array([]).reshape(0, 0)  # on retourne toujours un tableau 2D
        assert_equal(actual, expected)

        actual = exercices.get_center(np.arange(1, 10).reshape(3, 3))
        expected = [[5]]
        assert_equal(actual, expected)

    def test_swap_first_rows(self):
        # l'exemple doit marcher
        test_arr = np.array([[1, 2], [3, 4], [5, 6]])
        actual = exercices.swap_first_rows(test_arr)
        expected = [[3, 4], [1, 2], [5, 6]]
        assert_equal(actual, expected)

        # # cas aux limites
        test_arr = np.array([[1], [2]])
        actual = exercices.swap_first_rows(test_arr)
        expected = [[2], [1]]
        assert_equal(actual, expected)

        actual = exercices.swap_first_rows(np.array([[], []]))
        expected = [[], []]
        assert_equal(actual, expected)

    def test_funny_checkerboard(self):
        # l'exemple doit marcher
        actual = exercices.funny_checkerboard(5)
        expected = np.array(
            [
                # fmt: off
            [1, 0, 1, 0, 1,],
            [0, 1, 0, 1, 0,],
            [3, 0, 3, 0, 3,],
            [0, 1, 0, 1, 0,],
            [5, 0, 5, 0, 5,]
                # fmt: on
            ]
        )
        assert_equal(actual, expected)

        # cas aux limites
        actual = exercices.funny_checkerboard(1)
        expected = np.array([[1]])
        assert_equal(actual, expected)

        actual = exercices.funny_checkerboard(2)
        expected = np.array([[1, 0], [0, 1]])
        assert_equal(actual, expected)


class TestNiveau4:
    def test_sum_odd_columns(self):
        # l'exemple doit marcher
        test_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual = exercices.sum_odd_columns(test_arr)
        expected = 15
        assert_equal(actual, expected)

        # cas aux limites
        test_arr = np.array([[1, 2], [3, 4]])
        actual = exercices.sum_odd_columns(test_arr)
        expected = 6
        assert_equal(actual, expected)

        actual = exercices.sum_odd_columns(np.array([[1]]))
        expected = 0
        assert_equal(actual, expected)

    def test_mean(self):
        # l'exemple doit marcher
        test_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual = exercices.mean(test_arr)
        expected = 5
        assert_equal(actual, expected)

        # cas aux limites
        actual = exercices.mean(np.array([[1]]))
        expected = 1
        assert_equal(actual, expected)

        actual = exercices.mean(np.array([[0]]))
        expected = 0
        assert_equal(actual, expected)

    def test_max_per_line(self):
        # l'exemple doit marcher
        test_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual = exercices.max_per_line(test_arr)
        expected = [3, 6, 9]
        assert_equal(actual, expected)

        test_arr = np.array([[1, 3, 2], [6, 5, 4], [7, 8, 9]])
        actual = exercices.max_per_line(test_arr)
        expected = [3, 6, 9]
        assert_equal(actual, expected)

        # cas aux limites
        actual = exercices.max_per_line(np.array([[1]]))
        expected = [1]
        assert_equal(actual, expected)

    def test_min_per_column(self):
        # l'exemple doit marcher
        test_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        actual = exercices.min_per_column(test_arr)
        expected = [1, 2, 3]
        assert_equal(actual, expected)

        test_arr = np.array([[1, 3, 2], [6, 5, 4], [7, 8, 9]])
        actual = exercices.min_per_column(test_arr)
        expected = [1, 3, 2]
        assert_equal(actual, expected)

        # cas aux limites
        actual = exercices.min_per_column(np.array([[1]]))
        expected = [1]
        assert_equal(actual, expected)
