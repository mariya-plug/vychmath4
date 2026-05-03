"""
Лабораторная работа №4: Численное дифференцирование
Вариант 10
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

VARIANT = 10

# ======================== ПУНКТ 1: ИССЛЕДОВАНИЕ ПОГРЕШНОСТИ ========================


def f_x(v, x):
    """Функция f(x) = v * x^2"""
    return v * x**2


def f_derivative_exact(v, x):
    """Точная производная: f'(x) = 2*v*x"""
    return 2 * v * x


def forward_difference(f, x, h):
    """Формула (4.1): Правая разностная производная, порядок O(h)"""
    return (f(x + h) - f(x)) / h


def backward_difference(f, x, h):
    """Формула (4.2): Левая разностная производная, порядок O(h)"""
    return (f(x) - f(x - h)) / h


def central_difference(f, x, h):
    """Формула (4.3): Центральная разностная производная, порядок O(h^2)"""
    return (f(x + h) - f(x - h)) / (2 * h)


def analyze_derivative_accuracy():
    """
    Пункт 1: Вычислить производную функции f(x) = v*x^2 (v=10)
    в точках 10^{-2}, 10^{-1}, 1, 10, 10^2 тремя формулами при разных h.
    Построить графики зависимости погрешности от h.
    """
    print("=" * 90)
    print("ПУНКТ 1: ИССЛЕДОВАНИЕ ПОГРЕШНОСТИ ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ")
    print("=" * 90)
    print(f"Вариант: {VARIANT}")
    print(f"Функция: f(x) = {VARIANT} * x^2")
    print(f"Точная производная: f'(x) = {2 * VARIANT} * x")
    print()

    v = VARIANT
    func = lambda x: f_x(v, x)

    points = [1e-2, 1e-1, 1, 10, 1e2]
    h_values = np.logspace(0, -15, 61)

    eps_machine = np.finfo(float).eps  # ≈ 2.22e-16

    print(f"Машинная точность (epsilon float64): {eps_machine:.3e}")
    print()

    optimal_steps = {}

    all_results = {}

    for point in points:
        exact_val = f_derivative_exact(v, point)
        print(f"\n--- Точка x = {point:.0e}, f'(x) = {exact_val:.6e} ---")
        print(f"{'h':>12s}  {'Левая':>16s}  {'|ошибка|':>10s}  "
              f"{'Центральная':>16s}  {'|ошибка|':>10s}  "
              f"{'Правая':>16s}  {'|ошибка|':>10s}")
        print("-" * 102)

        errors_left = []
        errors_central = []
        errors_right = []

        h_display = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15]

        for h in h_values:
            al = backward_difference(func, point, h)
            ac = central_difference(func, point, h)
            ar = forward_difference(func, point, h)

            el = abs(al - exact_val)
            ec = abs(ac - exact_val)
            er = abs(ar - exact_val)

            errors_left.append(el)
            errors_central.append(ec)
            errors_right.append(er)

            if any(abs(h - hd) / max(hd, 1e-20) < 0.01 for hd in h_display):
                print(f"{h:12.1e}  {al:16.8e}  {el:10.2e}  "
                      f"{ac:16.8e}  {ec:10.2e}  "
                      f"{ar:16.8e}  {er:10.2e}")

        errors_left = np.array(errors_left)
        errors_central = np.array(errors_central)
        errors_right = np.array(errors_right)

        idx_best_left = np.argmin(errors_left)
        idx_best_cent = np.argmin(errors_central)
        idx_best_right = np.argmin(errors_right)

        print(f"\n  Оптимальный h (левая):       {h_values[idx_best_left]:.2e}, "
              f"ошибка = {errors_left[idx_best_left]:.2e}")
        print(f"  Оптимальный h (центральная): {h_values[idx_best_cent]:.2e}, "
              f"ошибка = {errors_central[idx_best_cent]:.2e}")
        print(f"  Оптимальный h (правая):      {h_values[idx_best_right]:.2e}, "
              f"ошибка = {errors_right[idx_best_right]:.2e}")

        optimal_steps[point] = {
            'h_left': h_values[idx_best_left],
            'h_central': h_values[idx_best_cent],
            'h_right': h_values[idx_best_right],
        }

        all_results[point] = {
            'h': h_values,
            'err_left': errors_left,
            'err_central': errors_central,
            'err_right': errors_right,
        }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, point in enumerate(points):
        ax = axes[idx]
        res = all_results[point]
        mask_l = res['err_left'] > 0
        mask_c = res['err_central'] > 0
        mask_r = res['err_right'] > 0

        if np.any(mask_l):
            ax.loglog(res['h'][mask_l], res['err_left'][mask_l], 'b-',
                      label='Левая O(h)', linewidth=1.2)
        if np.any(mask_c):
            ax.loglog(res['h'][mask_c], res['err_central'][mask_c], 'r-',
                      label='Центральная O(h²)', linewidth=1.2)
        if np.any(mask_r):
            ax.loglog(res['h'][mask_r], res['err_right'][mask_r], 'g--',
                      label='Правая O(h)', linewidth=1.2)

        ax.set_xlabel('h')
        ax.set_ylabel('|Погрешность|')
        ax.set_title(f'x = {point:.0e}')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()

    axes[5].axis('off')
    axes[5].text(0.1, 0.5,
                 "Погрешность по трём формулам\n"
                 "для f(x)=10x² при разных h.\n\n"
                 f"Machine eps = {eps_machine:.2e}\n"
                 f"f''(x) = {2*v} = const\n"
                 f"f'''(x) = 0 (квадратичная)\n\n"
                 "Для центральной разности:\n"
                 "ошибка метода O(h²·f'''),\n"
                 "f'''=0 → ошибка = 0 + округление",
                 fontsize=10, verticalalignment='center',
                 transform=axes[5].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Пункт 1: Зависимость погрешности от шага h', fontsize=14)
    plt.tight_layout()
    plt.savefig('punkt1_error_vs_h.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nГрафик сохранён: punkt1_error_vs_h.png")

    print("\n" + "=" * 90)
    print("ТЕОРЕТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ ПУНКТА 1")
    print("=" * 90)
    print(f"""
    Функция f(x) = {v}·x², f'(x) = {2*v}·x, f''(x) = {2*v}, f'''(x) = 0.

    1. ЛЕВАЯ / ПРАВАЯ РАЗНОСТНАЯ ПРОИЗВОДНАЯ (формулы 4.1, 4.2)
       Погрешность метода: R = (h/2)·f''(ξ) = {v}·h.
       Для f(x)={v}x² погрешность метода линейна по h.
       При уменьшении h погрешность метода уменьшается, но при h < ~sqrt(ε/M₂)
       начинает доминировать погрешность округления.

       Оптимальный шаг: h_opt ≈ 2·sqrt(ε·|f(x)|/|f''(x)|) ≈ 2·sqrt(ε·|{v}x²|/{2*v})
       = 2·|x|·sqrt(ε/2).
       При x=1: h_opt ≈ 2·sqrt({eps_machine:.2e}/2) ≈ {2*np.sqrt(eps_machine/2):.2e}.

    2. ЦЕНТРАЛЬНАЯ РАЗНОСТНАЯ ПРОИЗВОДНАЯ (формула 4.3)
       Погрешность метода: R = (h²/6)·f'''(ξ).
       Поскольку f'''(x) ≡ 0 для квадратичного полинома, ошибка метода НУЛЕВАЯ.
       Остаётся только погрешность округления ≈ ε·|f(x)|/(h) + ε·|f'(x)|.
       Поэтому центральная разность даёт почти машинную точность при больших h
       и теряет точность только при очень малых h из-за катастрофической потери
       значащих цифр при вычитании близких чисел.

    3. ЗАВИСИМОСТЬ ОТ ТОЧКИ x:
       - Абсолютная погрешность пропорциональна |f(x)| и |f'(x)|,
         поэтому для больших x погрешность округления выше.
       - Для x=100: f(100) = {v*100**2:.0f}, и даже для центральной разности
         при малых h потеря точности значительна.
    """)

    return optimal_steps


# ======================== ПУНКТ 2: РАСЧЕТ КОЭФФИЦИЕНТОВ КУБИЧЕСКОГО СПЛАЙНА ========================

x_data = np.array([2, 3, 4, 6, 7], dtype=float)
y_data = np.array([-1, -6, 7, 8, 2], dtype=float)


def solve_tridiagonal_vectors(a_diag, b_diag, c_diag, d_vec):
    """
    Метод прогонки (Thomas algorithm) для трёхдиагональной системы.
    a_diag — нижняя диагональ (длина n-1, a_diag[0] не используется),
    b_diag — главная диагональ (длина n),
    c_diag — верхняя диагональ (длина n-1),
    d_vec  — правая часть (длина n).
    """
    n = len(d_vec)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)

    cp[0] = c_diag[0] / b_diag[0]
    dp[0] = d_vec[0] / b_diag[0]

    for i in range(1, n):
        m = b_diag[i] - a_diag[i - 1] * cp[i - 1] if i < n else b_diag[i]
        if i - 1 < len(a_diag):
            m = b_diag[i] - a_diag[i - 1] * cp[i - 1]
        else:
            m = b_diag[i]
        if i < n - 1:
            cp[i] = c_diag[i] / m
        dp[i] = (d_vec[i] - (a_diag[i - 1] if i - 1 < len(a_diag) else 0) * dp[i - 1]) / m

    x = np.zeros(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def calculate_cubic_spline_coeffs(x, y):
    """
    Расчёт коэффициентов естественного кубического сплайна.
    S_i(t) = a_i + b_i·t + c_i·t² + d_i·t³,  t = x − x_i,  i = 0..n−1.

    Естественные граничные условия: S''(x_0) = 0, S''(x_n) = 0  ⟹  c_0 = 0, c_n = 0.
    """
    n = len(x) - 1
    h = np.diff(x)

    a = y[:-1].copy()

    lower = np.zeros(n + 1)
    main = np.zeros(n + 1)
    upper = np.zeros(n + 1)
    rhs = np.zeros(n + 1)

    main[0] = 1.0
    rhs[0] = 0.0

    for i in range(1, n):
        lower[i] = h[i - 1]
        main[i] = 2.0 * (h[i - 1] + h[i])
        upper[i] = h[i]
        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    main[n] = 1.0
    rhs[n] = 0.0

    c_all = solve_tridiagonal_vectors(lower[1:], main, upper[:-1], rhs)

    d = np.zeros(n)
    b = np.zeros(n)
    for i in range(n):
        d[i] = (c_all[i + 1] - c_all[i]) / (3.0 * h[i])
        b[i] = (y[i + 1] - y[i]) / h[i] - (c_all[i + 1] + 2.0 * c_all[i]) * h[i] / 3.0

    c = c_all[:n]
    return a, b, c, d, h, c_all


def spline_eval(x_val, x_nodes, a, b, c, d):
    """Значение сплайна в точке x_val."""
    n = len(a)
    for i in range(n - 1, -1, -1):
        if x_val >= x_nodes[i]:
            t = x_val - x_nodes[i]
            return a[i] + b[i] * t + c[i] * t**2 + d[i] * t**3
    t = x_val - x_nodes[0]
    return a[0] + b[0] * t + c[0] * t**2 + d[0] * t**3


def spline_deriv1(x_val, x_nodes, b, c, d):
    """Первая производная сплайна S'(x) = b_i + 2c_i·t + 3d_i·t²."""
    n = len(b)
    for i in range(n - 1, -1, -1):
        if x_val >= x_nodes[i]:
            t = x_val - x_nodes[i]
            return b[i] + 2.0 * c[i] * t + 3.0 * d[i] * t**2
    t = x_val - x_nodes[0]
    return b[0] + 2.0 * c[0] * t + 3.0 * d[0] * t**2


def spline_deriv2(x_val, x_nodes, c, d):
    """Вторая производная сплайна S''(x) = 2c_i + 6d_i·t."""
    n = len(c)
    for i in range(n - 1, -1, -1):
        if x_val >= x_nodes[i]:
            t = x_val - x_nodes[i]
            return 2.0 * c[i] + 6.0 * d[i] * t
    t = x_val - x_nodes[0]
    return 2.0 * c[0] + 6.0 * d[0] * t


def manual_calculation_demo():
    """Пункт 2: Ручной расчёт коэффициентов кубического сплайна с выводом системы."""
    print("\n" + "=" * 90)
    print("ПУНКТ 2: РАСЧЁТ КОЭФФИЦИЕНТОВ КУБИЧЕСКОГО СПЛАЙНА (Вариант 10)")
    print("=" * 90)

    print("\nИсходные данные (таблица 3.2, вариант 10):")
    print(f"  x    = {x_data.tolist()}")
    print(f"  f(x) = {y_data.tolist()}")

    n = len(x_data) - 1
    h = np.diff(x_data)

    print(f"\n  Количество интервалов n = {n}")
    print("\n  Шаги h_i = x_{i+1} − x_i:")
    for i in range(n):
        print(f"    h_{i} = x_{i+1} − x_{i} = {x_data[i+1]} − {x_data[i]} = {h[i]}")

    print("\n--- Система уравнений для c_i ---")
    print("  Естественные граничные условия: c_0 = 0, c_4 = 0.")
    print("  Уравнения для внутренних узлов (i = 1, 2, 3):")

    for i in range(1, n):
        lhs_terms = []
        if h[i-1] != 0:
            lhs_terms.append(f"{h[i-1]:.0f}·c_{i-1}")
        lhs_terms.append(f"{2*(h[i-1]+h[i]):.0f}·c_{i}")
        if h[i] != 0:
            lhs_terms.append(f"{h[i]:.0f}·c_{i+1}")
        rhs_val = 3.0 * ((y_data[i+1] - y_data[i]) / h[i] -
                          (y_data[i] - y_data[i-1]) / h[i-1])
        print(f"    i={i}: {' + '.join(lhs_terms)} = {rhs_val:.4f}")

    print("\n  Подставляя c_0 = 0 и c_4 = 0:")

    A_sys = np.zeros((n - 1, n - 1))
    b_sys = np.zeros(n - 1)

    for i in range(1, n):
        row = i - 1
        if i - 1 >= 1:
            A_sys[row, row - 1] = h[i - 1]
        A_sys[row, row] = 2.0 * (h[i - 1] + h[i])
        if i + 1 <= n - 1:
            A_sys[row, row + 1] = h[i]
        b_sys[row] = 3.0 * ((y_data[i + 1] - y_data[i]) / h[i] -
                             (y_data[i] - y_data[i - 1]) / h[i - 1])

    print("\n  Матрица системы (A · [c_1, c_2, c_3]^T = rhs):")
    for row in range(n - 1):
        row_str = "  |"
        for col in range(n - 1):
            row_str += f" {A_sys[row, col]:8.2f}"
        row_str += f" |  | c_{row+1} |"
        if row == 0:
            row_str += "     |"
        elif row == n - 2:
            row_str += "     |"
        else:
            row_str += "  =  |"
        row_str += f" {b_sys[row]:10.4f} |"
        print(row_str)

    c_inner = np.linalg.solve(A_sys, b_sys)

    print(f"\n  Решение: c_1 = {c_inner[0]:.6f}, c_2 = {c_inner[1]:.6f}, "
          f"c_3 = {c_inner[2]:.6f}")

    a, b, c, d, hh, c_all = calculate_cubic_spline_coeffs(x_data, y_data)

    print("\n--- Коэффициенты сплайна ---")
    print(f"\n  a_i = f(x_i):")
    for i in range(n):
        print(f"    a_{i} = {a[i]:.6f}")

    print(f"\n  c_i (из системы, c_0={c_all[0]:.1f}, c_{n}={c_all[n]:.1f}):")
    for i in range(n):
        print(f"    c_{i} = {c[i]:.6f}")

    print(f"\n  d_i = (c_{{i+1}} − c_i) / (3·h_i):")
    for i in range(n):
        print(f"    d_{i} = ({c_all[i+1]:.6f} − {c_all[i]:.6f}) / "
              f"(3·{hh[i]:.0f}) = {d[i]:.6f}")

    print(f"\n  b_i = (y_{{i+1}}−y_i)/h_i − (c_{{i+1}}+2c_i)·h_i/3:")
    for i in range(n):
        print(f"    b_{i} = ({y_data[i+1]:.0f}−({y_data[i]:.0f}))/{hh[i]:.0f} − "
              f"({c_all[i+1]:.6f}+2·{c_all[i]:.6f})·{hh[i]:.0f}/3 = {b[i]:.6f}")

    print("\n--- ИТОГОВАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ ---")
    print(f"{'Интервал':<14} {'a_i':>12} {'b_i':>12} {'c_i':>12} {'d_i':>12}")
    print("-" * 66)
    for i in range(n):
        print(f"[{x_data[i]:.0f}; {x_data[i+1]:.0f}]        "
              f"{a[i]:12.6f} {b[i]:12.6f} {c[i]:12.6f} {d[i]:12.6f}")

    return a, b, c, d, hh, c_all


# ======================== ПУНКТ 3: ПРОГРАММНАЯ РЕАЛИЗАЦИЯ ========================


def plot_spline_and_derivatives(x_nodes, y_nodes, a, b, c, d, h):
    """Построение графиков сплайна, первой и второй производной."""
    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 500)
    y_spline = np.array([spline_eval(xi, x_nodes, a, b, c, d) for xi in x_fine])
    y_d1 = np.array([spline_deriv1(xi, x_nodes, b, c, d) for xi in x_fine])
    y_d2 = np.array([spline_deriv2(xi, x_nodes, c, d) for xi in x_fine])

    d1_at_nodes = np.array([spline_deriv1(xi, x_nodes, b, c, d) for xi in x_nodes])
    d2_at_nodes = np.array([spline_deriv2(xi, x_nodes, c, d) for xi in x_nodes])

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    axes[0].plot(x_fine, y_spline, 'b-', linewidth=2, label='Кубический сплайн S(x)')
    axes[0].scatter(x_nodes, y_nodes, color='red', s=100, zorder=5,
                    label='Узлы таблицы', marker='o', edgecolors='black', linewidths=1.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('S(x)')
    axes[0].set_title('Кубический сплайн (интерполяция)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_fine, y_d1, 'g-', linewidth=2, label="S'(x)")
    axes[1].scatter(x_nodes, d1_at_nodes, color='red', s=80, zorder=5,
                    label='Значения в узлах', marker='s', edgecolors='black', linewidths=1.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("S'(x)")
    axes[1].set_title('Первая производная кубического сплайна')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_fine, y_d2, 'r-', linewidth=2, label="S''(x)")
    axes[2].scatter(x_nodes, d2_at_nodes, color='blue', s=80, zorder=5,
                    label='Значения в узлах', marker='^', edgecolors='black', linewidths=1.5)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel("S''(x)")
    axes[2].set_title('Вторая производная кубического сплайна')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Пункт 3: Графики сплайна и его производных (Вариант 10)', fontsize=14)
    plt.tight_layout()
    plt.savefig('cubic_spline_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("График сохранён: cubic_spline_results.png")

    print("\n--- ПРОВЕРКА НЕПРЕРЫВНОСТИ ПРОИЗВОДНЫХ ВО ВНУТРЕННИХ УЗЛАХ ---")
    for i in range(1, len(x_nodes) - 1):
        eps = 1e-10
        d1_left = spline_deriv1(x_nodes[i] - eps, x_nodes, b, c, d)
        d1_right = spline_deriv1(x_nodes[i] + eps, x_nodes, b, c, d)
        d2_left = spline_deriv2(x_nodes[i] - eps, x_nodes, c, d)
        d2_right = spline_deriv2(x_nodes[i] + eps, x_nodes, c, d)

        print(f"  Узел x = {x_nodes[i]:.0f}:")
        print(f"    S'(x−ε)  = {d1_left:.10f},  S'(x+ε)  = {d1_right:.10f},  "
              f"|разность| = {abs(d1_left - d1_right):.2e}")
        print(f"    S''(x−ε) = {d2_left:.10f},  S''(x+ε) = {d2_right:.10f},  "
              f"|разность| = {abs(d2_left - d2_right):.2e}")


def compare_with_scipy(x_nodes, y_nodes, a, b, c, d):
    """Сравнение собственной реализации с scipy.interpolate.CubicSpline."""
    cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')

    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 300)
    y_scipy = cs(x_fine)
    yd1_scipy = cs(x_fine, 1)
    yd2_scipy = cs(x_fine, 2)

    y_ours = np.array([spline_eval(xi, x_nodes, a, b, c, d) for xi in x_fine])
    yd1_ours = np.array([spline_deriv1(xi, x_nodes, b, c, d) for xi in x_fine])
    yd2_ours = np.array([spline_deriv2(xi, x_nodes, c, d) for xi in x_fine])

    err_s = np.max(np.abs(y_scipy - y_ours))
    err_d1 = np.max(np.abs(yd1_scipy - yd1_ours))
    err_d2 = np.max(np.abs(yd2_scipy - yd2_ours))

    print("\n--- СРАВНЕНИЕ С SCIPY (CubicSpline, bc_type='natural') ---")
    print(f"  max|S(x) − S_scipy(x)|   = {err_s:.2e}")
    print(f"  max|S'(x) − S'_scipy(x)| = {err_d1:.2e}")
    print(f"  max|S''(x)−S''_scipy(x)| = {err_d2:.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(x_fine, y_scipy, 'b-', label='scipy', linewidth=2)
    axes[0].plot(x_fine, y_ours, 'r--', label='Наша реализация', linewidth=2)
    axes[0].scatter(x_nodes, y_nodes, color='black', s=60, zorder=5, label='Узлы')
    axes[0].set_title('S(x): сравнение')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_fine, yd1_scipy, 'b-', label='scipy', linewidth=2)
    axes[1].plot(x_fine, yd1_ours, 'r--', label='Наша реализация', linewidth=2)
    axes[1].set_title("S'(x): сравнение")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_fine, yd2_scipy, 'b-', label='scipy', linewidth=2)
    axes[2].plot(x_fine, yd2_ours, 'r--', label='Наша реализация', linewidth=2)
    axes[2].set_title("S''(x): сравнение")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Сравнение с scipy.interpolate.CubicSpline', fontsize=13)
    plt.tight_layout()
    plt.savefig('comparison_with_scipy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  График сохранён: comparison_with_scipy.png")


def numerical_derivatives_at_table_nodes(x_nodes, y_nodes, b, c, d, h):
    """
    Пункт 3 (дополнительно): Численные производные в узлах таблицы
    по разностным формулам (4.1)–(4.3), учитывая НЕРАВНОМЕРНЫЙ шаг.
    Сравнение с производной сплайна.
    """
    print("\n--- ЧИСЛЕННЫЕ ПРОИЗВОДНЫЕ В УЗЛАХ ТАБЛИЦЫ ---")
    print("  (с учётом неравномерного шага между узлами)")
    n = len(x_nodes)

    print(f"\n  {'i':<4} {'x_i':<6} {'f(x_i)':<8} {'Числ.произв.':<16} "
          f"{'Произв.сплайна':<16} {'|Разность|':<12} {'Метод'}")
    print("  " + "-" * 80)

    for i in range(n):
        s_deriv = spline_deriv1(x_nodes[i], x_nodes, b, c, d)

        if i == 0:
            hi = x_nodes[1] - x_nodes[0]
            num_d = (y_nodes[1] - y_nodes[0]) / hi
            method = "правая (4.1)"
        elif i == n - 1:
            hi = x_nodes[-1] - x_nodes[-2]
            num_d = (y_nodes[-1] - y_nodes[-2]) / hi
            method = "левая (4.2)"
        else:
            h_left = x_nodes[i] - x_nodes[i - 1]
            h_right = x_nodes[i + 1] - x_nodes[i]
            if abs(h_left - h_right) < 1e-12:
                num_d = (y_nodes[i + 1] - y_nodes[i - 1]) / (h_left + h_right)
                method = "центр. (4.3)"
            else:
                num_d = (
                    -h_right / (h_left * (h_left + h_right)) * y_nodes[i - 1]
                    + (h_right - h_left) / (h_left * h_right) * y_nodes[i]
                    + h_left / (h_right * (h_left + h_right)) * y_nodes[i + 1]
                )
                method = f"неравн. h_l={h_left:.0f} h_r={h_right:.0f}"

        diff = abs(num_d - s_deriv)
        print(f"  {i:<4} {x_nodes[i]:<6.1f} {y_nodes[i]:<8.1f} "
              f"{num_d:<16.6f} {s_deriv:<16.6f} {diff:<12.2e} {method}")


# ======================== ОСНОВНАЯ ФУНКЦИЯ ========================


def main():
    print("╔" + "═" * 88 + "╗")
    print("║  ЛАБОРАТОРНАЯ РАБОТА №4: ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ".ljust(89) + "║")
    print("║  Вариант 10".ljust(89) + "║")
    print("╚" + "═" * 88 + "╝")

    optimal_steps = analyze_derivative_accuracy()

    a, b, c, d, h, c_all = manual_calculation_demo()

    print("\n" + "=" * 90)
    print("ПУНКТ 3: ПРОГРАММНАЯ РЕАЛИЗАЦИЯ — ГРАФИКИ И ПРОВЕРКИ")
    print("=" * 90)

    plot_spline_and_derivatives(x_data, y_data, a, b, c, d, h)
    compare_with_scipy(x_data, y_data, a, b, c, d)
    numerical_derivatives_at_table_nodes(x_data, y_data, b, c, d, h)

    print("\n" + "=" * 90)
    print("ПРОВЕРКА СВОЙСТВ СПЛАЙНА")
    print("=" * 90)

    print("\n  1. Интерполяция (S(x_i) = f(x_i)):")
    for i, xi in enumerate(x_data):
        si = spline_eval(xi, x_data, a, b, c, d)
        print(f"     S({xi:.0f}) = {si:.10f},  f(x_{i}) = {y_data[i]:.1f},  "
              f"|разность| = {abs(si - y_data[i]):.2e}")

    print("\n  2. Естественные граничные условия S''(x_0)=0, S''(x_n)=0:")
    d2_left = spline_deriv2(x_data[0], x_data, c, d)
    d2_right = spline_deriv2(x_data[-1], x_data, c, d)
    print(f"     S''({x_data[0]:.0f}) = {d2_left:.10f}  (должно быть 0)")
    print(f"     S''({x_data[-1]:.0f}) = {d2_right:.10f}  (должно быть 0)")

    print("\n" + "=" * 90)
    print("ВЫВОДЫ ПО РАБОТЕ")
    print("=" * 90)
    print(f"""
    1. Пункт 1 (исследование погрешности):
       - Для f(x) = {VARIANT}·x² центральная разность (формула 4.3) даёт точный
         результат при умеренных h, т.к. f'''(x) ≡ 0 (погрешность метода нулевая).
       - Левая и правая разности (формулы 4.1, 4.2) имеют погрешность O(h),
         линейно убывающую с уменьшением h до тех пор, пока не начнёт
         доминировать погрешность округления.
       - При очень малых h (< 10⁻⁸) все формулы теряют точность из-за
         катастрофической потери значащих цифр при вычитании близких чисел.

    2. Пункт 2 (расчёт коэффициентов сплайна):
       - Построен естественный кубический сплайн по 5 узлам [{', '.join(str(int(xi)) for xi in x_data)}].
       - Коэффициенты получены решением трёхдиагональной системы методом прогонки.
       - Результат совпадает с scipy.interpolate.CubicSpline (bc_type='natural').

    3. Пункт 3 (графики и проверки):
       - Сплайн точно проходит через все узлы таблицы.
       - Первая и вторая производные непрерывны во внутренних узлах.
       - Естественные граничные условия выполнены: S''(x_0)=S''(x_n)=0.
       - Численные производные по разностным формулам в узлах сравнены
         с производной сплайна.
    """)


if __name__ == "__main__":
    main()
