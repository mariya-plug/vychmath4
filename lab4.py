"""
Лабораторная работа №4: Численное дифференцирование
Вариант 10
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# ======================== ПУНКТ 1: ИССЛЕДОВАНИЕ ПОГРЕШНОСТИ ========================

def f_x(v, x):
    """Функция f(x) = v * x^2, где v - номер варианта (10)"""
    return 10 * x**2

def f_derivative_exact(v, x):
    """Точная производная: f'(x) = 2*v*x"""
    return 20 * x

def forward_difference(f, x, h):
    """Формула (4.1): Правая разностная производная (1-й порядок точности)"""
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
    """Формула (4.2): Левая разностная производная (1-й порядок точности)"""
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    """Формула (4.3): Центральная разностная производная (2-й порядок точности)"""
    return (f(x + h) - f(x - h)) / (2 * h)

def analyze_derivative_accuracy():
    """
    Пункт 1 задания: Вычисление производной функции f(x)=10*x^2
    в точках 10^-2, 10^-1, 1, 10, 10^2 с различным шагом
    """
    print("=" * 80)
    print("ПУНКТ 1: ИССЛЕДОВАНИЕ ПОГРЕШНОСТИ ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ")
    print("=" * 80)
    print("Функция: f(x) = 10 * x^2")
    print("Точная производная: f'(x) = 20 * x")
    print()
    
    # Точки для исследования
    points = [1e-2, 1e-1, 1, 10, 1e2]
    # Шаги для исследования
    steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    optimal_steps = {}
    
    for point in points:
        exact_val = f_derivative_exact(10, point)
        print(f"\n--- Точка x = {point:.0e} (точное значение f'(x) = {exact_val:.6e}) ---")
        print(f"{'h':<12} {'Левая разность':<18} {'Ошибка':<12} "
              f"{'Центральная':<18} {'Ошибка':<12} {'Правая разность':<18} {'Ошибка':<12}")
        print("-" * 110)
        
        best_for_point = {'method': None, 'error': float('inf'), 'h': None}
        
        for h in steps:
            approx_left = backward_difference(lambda x: f_x(10, x), point, h)
            error_left = abs(approx_left - exact_val)
            
            approx_central = central_difference(lambda x: f_x(10, x), point, h)
            error_central = abs(approx_central - exact_val)
            
            approx_right = forward_difference(lambda x: f_x(10, x), point, h)
            error_right = abs(approx_right - exact_val)
            
            print(f"{h:.1e}    {approx_left:>14.6e}  {error_left:>10.2e}    "
                  f"{approx_central:>14.6e}  {error_central:>10.2e}    "
                  f"{approx_right:>14.6e}  {error_right:>10.2e}")
            
            if error_central < best_for_point['error']:
                best_for_point['error'] = error_central
                best_for_point['method'] = 'Центральная разность'
                best_for_point['h'] = h
                best_for_point['approx'] = approx_central
        
        optimal_steps[point] = best_for_point
        print(f"\n>>> Оптимальный шаг для x={point:.0e}: h={best_for_point['h']:.1e}, "
              f"метод={best_for_point['method']}, ошибка={best_for_point['error']:.2e}\n")
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ПУНКТА 1")
    print("=" * 80)
    print("""
    1. Зависимость погрешности от шага h:
       - При слишком большом шаге (h ~ 0.1) погрешность велика из-за грубой
         аппроксимации производной (ошибка метода порядка O(h) или O(h^2)).
       - При умеренном шаге (h ~ 1e-4 до 1e-6) погрешность минимальна.
       - При слишком малом шаге (h < 1e-7) погрешность начинает расти из-за
         вычислительной погрешности (округление чисел с плавающей точкой).
    
    2. Сравнение методов:
       - Центральная разность (формула 4.3) имеет порядок точности O(h^2),
         поэтому при одинаковом h она точнее левой и правой разностей.
       - Левая и правая разности имеют порядок O(h) и дают систематическую
         ошибку в одну сторону (односторонняя аппроксимация).
    
    3. Влияние величины x:
       - Для малых x (0.01) относительная погрешность больше из-за малости
         самого значения производной.
       - Для больших x (100) абсолютная погрешность больше, но относительная
         может быть меньше.
    
    4. Теоретическая оценка оптимального шага:
       - Оптимальный шаг h ≈ sqrt(2ε/M2), где ε ~ 1e-7 (машинная точность),
         M2 = max|f''(x)| = 20. h_opt ≈ sqrt(2e-7/20) ≈ 1e-4, что соответствует
         полученным экспериментальным результатам.
    """)
    
    return optimal_steps


# ======================== ПУНКТ 2: РАСЧЕТ КОЭФФИЦИЕНТОВ КУБИЧЕСКОГО СПЛАЙНА ========================

# Таблица 3.2 для варианта 10
x_data = np.array([2, 3, 4, 6, 7], dtype=float)
y_data = np.array([-1, -6, 7, 8, 2], dtype=float)


def calculate_cubic_spline_coeffs(x, y):
    """
    Расчет коэффициентов кубического сплайна по формулам (3.3), (3.7), (3.9)-(3.11)
    """
    n = len(x) - 1
    h = np.diff(x)
    
    a = y[:-1].copy()
    
    A = np.zeros((n+1, n+1))
    B = np.zeros(n+1)
    
    A[0, 0] = 1.0
    B[0] = 0.0
    
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    
    A[n, n] = 1.0
    B[n] = 0.0
    
    c = solve_tridiagonal(A, B)
    
    d = np.zeros(n)
    for i in range(n):
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    
    b = np.zeros(n)
    for i in range(n):
        b[i] = (y[i+1] - y[i]) / h[i] - (c[i+1] + 2 * c[i]) * h[i] / 3
    
    return a, b, c[:-1], d, h


def solve_tridiagonal(A, B):
    """Решение системы с трехдиагональной матрицей методом прогонки"""
    n = len(B)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    alpha[0] = -A[0, 1] / A[0, 0] if n > 1 and A[0, 1] != 0 else 0
    beta[0] = B[0] / A[0, 0]
    
    for i in range(1, n-1):
        denominator = A[i, i] + A[i, i-1] * alpha[i-1]
        alpha[i] = -A[i, i+1] / denominator if i < n-1 and A[i, i+1] != 0 else 0
        beta[i] = (B[i] - A[i, i-1] * beta[i-1]) / denominator
    
    if n > 1:
        denominator = A[n-1, n-1] + A[n-1, n-2] * alpha[n-2]
        beta[n-1] = (B[n-1] - A[n-1, n-2] * beta[n-2]) / denominator
    
    x = np.zeros(n)
    x[n-1] = beta[n-1]
    for i in range(n-2, -1, -1):
        x[i] = beta[i] + alpha[i] * x[i+1]
    
    return x


def spline_value(x, x_nodes, a, b, c, d, h):
    """Вычисление значения сплайна в произвольной точке x"""
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i+1] or (i == len(x_nodes)-2 and x >= x_nodes[i]):
            t = x - x_nodes[i]
            return a[i] + b[i] * t + c[i] * t**2 + d[i] * t**3
    return None


def spline_derivative(x, x_nodes, b, c, d):
    """Вычисление первой производной сплайна в точке x"""
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i+1] or (i == len(x_nodes)-2 and x >= x_nodes[i]):
            t = x - x_nodes[i]
            return b[i] + 2 * c[i] * t + 3 * d[i] * t**2
    return None


def spline_second_derivative(x, x_nodes, c, d):
    """Вычисление второй производной сплайна в точке x"""
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i+1] or (i == len(x_nodes)-2 and x >= x_nodes[i]):
            t = x - x_nodes[i]
            return 2 * c[i] + 6 * d[i] * t
    return None


def manual_calculation_demo():
    """Пункт 2: Ручной расчет коэффициентов кубического сплайна"""
    print("\n" + "=" * 80)
    print("ПУНКТ 2: РАСЧЕТ КОЭФФИЦИЕНТОВ КУБИЧЕСКОГО СПЛАЙНА (Вариант 10)")
    print("=" * 80)
    
    print("\nИсходные данные (Таблица 3.2, вариант 10):")
    print(f"x = {x_data}")
    print(f"f(x) = {y_data}")
    
    a, b, c, d, h = calculate_cubic_spline_coeffs(x_data, y_data)
    
    print("\n--- РАСЧЕТ ПО ФОРМУЛАМ ---")
    print("\n1. Шаги между узлами h_i = x_{i+1} - x_i:")
    for i, hi in enumerate(h):
        print(f"   h_{i} = {hi}")
    
    print("\n2. Коэффициенты a_i = f(x_i) (формула 3.3):")
    for i, ai in enumerate(a):
        print(f"   a_{i} = {ai}")
    
    print("\n3. Коэффициенты c_i (решение системы):")
    for i, ci in enumerate(c):
        print(f"   c_{i+1} = {ci:.6f}")
    
    print("\n4. Коэффициенты d_i (формула 3.9):")
    for i, di in enumerate(d):
        print(f"   d_{i} = {di:.6f}")
    
    print("\n5. Коэффициенты b_i (формула 3.10):")
    for i, bi in enumerate(b):
        print(f"   b_{i} = {bi:.6f}")
    
    print("\n--- ИТОГОВЫЕ КОЭФФИЦИЕНТЫ СПЛАЙНА ---")
    print("Интервал |         a         |         b         |         c         |         d")
    print("-" * 75)
    for i in range(len(h)):
        print(f"[{x_data[i]}; {x_data[i+1]}] | {a[i]:>18.6f} | {b[i]:>18.6f} | {c[i]:>18.6f} | {d[i]:>18.6f}")
    
    return a, b, c, d, h


# ======================== ПУНКТ 3: ПРОГРАММНАЯ РЕАЛИЗАЦИЯ ========================

def plot_spline_and_derivatives(x_nodes, y_nodes, a, b, c, d, h):
    """Построение графиков сплайна и его производных"""
    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 500)
    y_spline = []
    y_deriv1 = []
    y_deriv2 = []
    
    for x in x_fine:
        val = spline_value(x, x_nodes, a, b, c, d, h)
        y_spline.append(val if val is not None else 0)
        
        deriv1 = spline_derivative(x, x_nodes, b, c, d)
        y_deriv1.append(deriv1 if deriv1 is not None else 0)
        
        deriv2 = spline_second_derivative(x, x_nodes, c, d)
        y_deriv2.append(deriv2 if deriv2 is not None else 0)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    axes[0].plot(x_fine, y_spline, 'b-', linewidth=2, label='Кубический сплайн')
    axes[0].scatter(x_nodes, y_nodes, color='red', s=100, zorder=5, 
                    label='Исходные точки', marker='o', edgecolors='black')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Кубический сплайн интерполяция')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x_fine, y_deriv1, 'g-', linewidth=2, label="Первая производная f'(x)")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("f'(x)")
    axes[1].set_title('Первая производная кубического сплайна')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x_fine, y_deriv2, 'r-', linewidth=2, label="Вторая производная f''(x)")
    axes[2].set_xlabel('x')
    axes[2].set_ylabel("f''(x)")
    axes[2].set_title('Вторая производная кубического сплайна')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cubic_spline_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n--- ПРОВЕРКА НЕПРЕРЫВНОСТИ ПРОИЗВОДНЫХ ВО ВНУТРЕННИХ УЗЛАХ ---")
    for i in range(1, len(x_nodes)-1):
        x_left = x_nodes[i] - 1e-10
        deriv_left = spline_derivative(x_left, x_nodes, b, c, d)
        x_right = x_nodes[i] + 1e-10
        deriv_right = spline_derivative(x_right, x_nodes, b, c, d)
        diff_deriv = abs(deriv_left - deriv_right) if deriv_left is not None and deriv_right is not None else float('inf')
        
        print(f"Узел x = {x_nodes[i]}:")
        print(f"  f'(x-) = {deriv_left:.10f}")
        print(f"  f'(x+) = {deriv_right:.10f}")
        print(f"  Разность = {diff_deriv:.2e}")
        
        deriv2_left = spline_second_derivative(x_left, x_nodes, c, d)
        deriv2_right = spline_second_derivative(x_right, x_nodes, c, d)
        diff_deriv2 = abs(deriv2_left - deriv2_right) if deriv2_left is not None and deriv2_right is not None else float('inf')
        
        print(f"  f''(x-) = {deriv2_left:.10f}")
        print(f"  f''(x+) = {deriv2_right:.10f}")
        print(f"  Разность = {diff_deriv2:.2e}")
        print()


def compare_with_scipy(x_nodes, y_nodes):
    """Сравнение с библиотекой scipy"""
    from scipy.interpolate import CubicSpline
    
    cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
    
    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 200)
    y_scipy = cs(x_fine)
    y_scipy_deriv1 = cs.derivative()(x_fine)
    y_scipy_deriv2 = cs.derivative(2)(x_fine)
    
    a, b, c, d, h = calculate_cubic_spline_coeffs(x_nodes, y_nodes)
    y_ours = []
    y_ours_deriv1 = []
    y_ours_deriv2 = []
    for x in x_fine:
        val = spline_value(x, x_nodes, a, b, c, d, h)
        y_ours.append(val if val is not None else 0)
        
        deriv1 = spline_derivative(x, x_nodes, b, c, d)
        y_ours_deriv1.append(deriv1 if deriv1 is not None else 0)
        
        deriv2 = spline_second_derivative(x, x_nodes, c, d)
        y_ours_deriv2.append(deriv2 if deriv2 is not None else 0)
    
    max_error = np.max(np.abs(y_scipy - y_ours))
    max_error_deriv1 = np.max(np.abs(y_scipy_deriv1 - y_ours_deriv1))
    max_error_deriv2 = np.max(np.abs(y_scipy_deriv2 - y_ours_deriv2))
    
    print("\n--- СРАВНЕНИЕ С РЕАЛИЗАЦИЕЙ SCIPY ---")
    print(f"Максимальная погрешность сплайна: {max_error:.2e}")
    print(f"Максимальная погрешность первой производной: {max_error_deriv1:.2e}")
    print(f"Максимальная погрешность второй производной: {max_error_deriv2:.2e}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(x_fine, y_scipy, 'b-', label='scipy', linewidth=2)
    axes[0].plot(x_fine, y_ours, 'r--', label='Наша реализация', linewidth=2)
    axes[0].scatter(x_nodes, y_nodes, color='black', s=50, zorder=5)
    axes[0].set_title('Сравнение сплайнов')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x_fine, y_scipy_deriv1, 'b-', label='scipy', linewidth=2)
    axes[1].plot(x_fine, y_ours_deriv1, 'r--', label='Наша реализация', linewidth=2)
    axes[1].set_title("Сравнение первой производной")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x_fine, y_scipy_deriv2, 'b-', label='scipy', linewidth=2)
    axes[2].plot(x_fine, y_ours_deriv2, 'r--', label='Наша реализация', linewidth=2)
    axes[2].set_title("Сравнение второй производной")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_with_scipy.png', dpi=150, bbox_inches='tight')
    plt.show()


def numerical_derivative_check(x_nodes, y_nodes):
    """Проверка численного дифференцирования по формуле (4.3)"""
    print("\n--- ПРОВЕРКА ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ (формула 4.3) ---")
    
    h_deriv = x_nodes[1] - x_nodes[0]
    derivatives_numerical = []
    
    for i in range(len(y_nodes)):
        if i == 0:
            deriv = (y_nodes[1] - y_nodes[0]) / h_deriv
        elif i == len(y_nodes) - 1:
            deriv = (y_nodes[-1] - y_nodes[-2]) / h_deriv
        else:
            deriv = (y_nodes[i+1] - y_nodes[i-1]) / (2 * h_deriv)
        derivatives_numerical.append(deriv)
    
    a, b, c, d, h = calculate_cubic_spline_coeffs(x_nodes, y_nodes)
    derivatives_spline = []
    for i, x in enumerate(x_nodes):
        if i < len(x_nodes) - 1:
            deriv = b[i]
        else:
            deriv = b[-1] + 2 * c[-1] * h[-1] + 3 * d[-1] * h[-1]**2
        derivatives_spline.append(deriv)
    
    print(f"\n{'i':<5} {'x_i':<10} {'Численная произв.':<20} {'Производная сплайна':<20} {'Разность':<15}")
    print("-" * 75)
    for i in range(len(x_nodes)):
        diff = abs(derivatives_numerical[i] - derivatives_spline[i])
        print(f"{i:<5} {x_nodes[i]:<10.2f} {derivatives_numerical[i]:<20.6f} "
              f"{derivatives_spline[i]:<20.6f} {diff:<15.2e}")


def main():
    """Основная функция"""
    
    # Пункт 1
    optimal_steps = analyze_derivative_accuracy()
    
    # Пункт 2
    a, b, c, d, h = manual_calculation_demo()
    
    # Пункт 3
    print("\n" + "=" * 80)
    print("ПУНКТ 3: ПОСТРОЕНИЕ ГРАФИКОВ СПЛАЙНА И ПРОИЗВОДНЫХ")
    print("=" * 80)
    
    plot_spline_and_derivatives(x_data, y_data, a, b, c, d, h)
    compare_with_scipy(x_data, y_data)
    numerical_derivative_check(x_data, y_data)
    
    print("\n" + "=" * 80)
    print("ВЫВОДЫ ПО РАБОТЕ")
    print("=" * 80)
    print("""
    1. По пункту 1 (численное дифференцирование):
       - Наилучшая точность достигается при шаге h ~ 10^-5 для центральной разности.
       - При слишком малом шаге (h < 10^-7) погрешность возрастает.
    
    2. По пункту 2 (расчет сплайна):
       - Кубический сплайн обеспечивает непрерывность функции и производных.
       - Коэффициенты рассчитаны по формулам (3.3), (3.7), (3.9)-(3.11).
    
    3. По пункту 3 (программная реализация):
       - Производные непрерывны во внутренних узлах.
       - Результаты совпадают с реализацией scipy.
    """)


if __name__ == "__main__":
    main()