import numpy as np
import matplotlib.pyplot as plt

# Граф интенсивностей
GRAPH = [
    [0, 2, 5, 3, 9],
    [0, 0, 0, 8, 12],
    [0, 1, 0, 0, 2],
    [4, 6, 7, 0, 0],
    [0, 3, 0, 6, 0]
]

N = len(GRAPH)
L = np.array(GRAPH, dtype=float)

# Функция создания матрицы A (для уравнений Колмогорова)
def build_kolmogorov_system(L):
    # Создаем матрицу A размером NxN, заполненную нулями
    A = np.zeros_like(L)
    for i in range(N):
        for j in range(N):
            if i != j:  # Проверяем, что не обрабатываем диагональные элементы
                A[i, j] = L[j, i]  # Заполняем входящие переходы
                A[i, i] -= L[i, j]  # Вычитаем исходящие переходы для диагонального элемента
    return A

# Метод Эйлера для решения системы ДУ
def euler_method(L, P0, dt, t_max):
    steps = int(t_max / dt)  # Количество шагов интегрирования
    solutions = []  # Список для хранения значений вероятностей на каждом шаге
    P = P0.copy()  # Копируем начальные условия
    solutions.append(P.copy())  # Сохраняем начальное состояние

    for _ in range(steps):  # Итерации по времени
        dP = np.zeros(N)  # Изменение вероятностей на текущем шаге
        for i in range(N):
            for j in range(N):
                if i != j:  # Проверяем, что не обрабатываем диагональные элементы
                    dP[i] -= L[i, j] * P[i]  # Вычитаем исходящие переходы
                    dP[i] += L[j, i] * P[j]  # Добавляем входящие переходы

        P += dP * dt  # Обновляем вероятности
        P = np.clip(P, 0, 1)  # Убеждаемся, что вероятности остаются в пределах [0, 1]
        solutions.append(P.copy())  # Сохраняем состояние после текущего шага

    return solutions

# Функция проверки эргодичности системы
def is_ergodic(L):
    def dfs(start, graph):
        visited = set()
        stack = [start]
        while stack:
            state = stack.pop()
            if state not in visited:
                visited.add(state)
                for next_state, weight in enumerate(graph[state]):
                    if weight > 0:
                        stack.append(next_state)
        return visited

    # Прямая проверка достижимости из каждого состояния
    for i in range(N):
        if len(dfs(i, L)) < N:  # Если из i нельзя достигнуть всех состояний
            return False

    # Обратная проверка достижимости (транспонированный граф)
    LT = L.T  # Транспонирование графа
    for i in range(N):
        if len(dfs(i, LT)) < N:  # Если в i нельзя попасть из всех состояний
            return False

    return True

# Функция нахождения предельных вероятностей
def find_steady_state(A):
    A_original = A.copy()  # Сохраняем исходную матрицу для вывода
    print("Матрица A до модификации:")
    print(A_original)  # Печать матрицы A до модификации
    b = np.zeros(N)  # Вектор правой части
    b[-1] = 1  # Условие нормировки
    print("\nСтолбец свободных членов:", b)  # Печать столбца свободных членов
    A[-1] = np.ones(N)  # Заменяем последнюю строку на условие нормировки (сумма вероятностей = 1)
    return np.linalg.solve(A, b)  # Решаем линейную систему уравнений

dt = 0.01  # Шаг времени
T_MAX = 2  # Максимальное время симуляции
P0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Начальное состояние: система находится в состоянии 1

A = build_kolmogorov_system(L)  # Построение матрицы уравнений Колмогорова
solutions = euler_method(L, P0, dt, T_MAX)  # Решение методом Эйлера

ergodic = is_ergodic(L)  # Проверяем, является ли граф эргодичным

steady_state = find_steady_state(A)  # Вычисляем предельные вероятности

print("Матрица A (система Колмогорова):")
print(A)  # Печать матрицы A после модификации
print("\nЭргодична ли система:", "Да" if ergodic else "Нет")
print("\nПредельные вероятности:")
for i, p in enumerate(steady_state):
    print(f"P{i+1} = {p:.4f}")

# Построение графиков переходных вероятностей
solutions = np.array(solutions)
time = np.linspace(0, T_MAX, len(solutions))

plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(time, solutions[:, i], label=f'State S{i+1}')  # Строим график для каждого состояния

plt.xlabel('Time')  # Ось X
plt.ylabel('Probability')  # Ось Y
plt.legend()
plt.grid()
plt.show()
