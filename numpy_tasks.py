import numpy as np

def uniform_intervals(a, b, n):
    """Создает numpy массив - равномерное разбиение интервала от a до b на n отрезков.

    Args:
        a: Начало интервала.
        b: Конец интервала.
        n: Количество отрезков (а также количество точек в массиве,
           включая обе границы).

    Returns:
        numpy.ndarray: Массив numpy, содержащий точки равномерного разбиения.
                       Первый элемент - a, последний - b.
    """
    return np.linspace(a, b, n)

def test1():
    assert np.allclose(uniform_intervals(-1.2, 2.4, 7), np.array([-1.2, -0.6,  0. ,  0.6,  1.2,  1.8,  2.4]))

# Запуск тестов
test1()
print("Тест пройден!")

def cyclic123_array(n):
    """Генерирует numpy массив длины 3n, заполненный циклически числами 1, 2, 3, 1, 2, 3, 1....

    Args:
        n: Количество повторений цикла (1, 2, 3).  Результирующая длина массива будет 3*n.

    Returns:
        numpy.ndarray: Массив numpy, содержащий циклически повторяющиеся числа 1, 2, 3.
    """
    base_array = np.array([1, 2, 3])
    return np.tile(base_array, n)

def test2():
    assert np.allclose(cyclic123_array(4), np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]))

# Запуск тестов
test2()
print("Тест 2 пройден!")

def first_n_odd_number(n):
    """3. Создает массив первых n нечетных целых чисел"""
    pass

def zeros_array_with_border(n):
    """4. Создает массив нулей размера n x n с "рамкой" из единиц по краям."""
    pass

def chess_board(n):
    """5. Создаёт массив n x n с шахматной доской из нулей и единиц"""
    pass

def matrix_with_sum_index(n):
    """6. Создаёт 𝑛 × 𝑛  матрицу с (𝑖,𝑗)-элементами равным 𝑖+𝑗."""
    pass

def cos_sin_as_two_rows(a, b, dx):
    """7. Вычислите $cos(x)$ и $sin(x)$ на интервале [a, b) с шагом dx, 
    а затем объедините оба массива чисел как строки в один массив. """
    pass

def compute_mean_rowssum_columnssum(A):
    """8. Для numpy массива A вычисляет среднее всех элементов, сумму строк и сумму столбцов."""
    pass

def sort_array_by_column(A, j):
    """ 9. Сортирует строки numpy массива A по j-му столбцу в порядке возрастания."""
    pass

def compute_integral(a, b, f, dx, method):
    """10. Считает определённый интеграл функции f на отрезке [a, b] с шагом dx 3-мя методами:  
    method == 'rectangular' - методом прямоугольника   
    method == 'trapezoidal' - методом трапеций   
    method == 'simpson' - методом Симпсона  
    """
    pass
