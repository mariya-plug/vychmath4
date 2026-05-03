"""
Лабораторная работа №4: Численное дифференцирование
Вариант 10

Методичка: Ландовский В.В. «Численные методы», НГТУ, 2022
Раздел 4.2, задание к лабораторной работе
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

VARIANT = 10

# ======================== ПУНКТ 1 ========================
# "Запрограммировать приближенные вычисления первой производной функции
#  f(x) = v·x², где v — номер варианта, в точках 10^-2, 10^-1, 1, 10, 10^2.
#  В программе использовать 32 битное представление вещественных чисел.
#  Сделать вычисления с различным шагом: 10^-1, 10^-2, ..., 10^-8.
#  Экспериментально подобрать оптимальный шаг."
# ==========================================================

F32 = np.float32


def f_x(x):
    """f(x) = 10·x²  (float32)"""
    return F32(VARIANT) * x * x


def f_deriv_exact(x):
    """Точная производная f'(x) = 20·x  (float64 для сравнения)"""
    return 2.0 * VARIANT * float(x)


def forward_diff(f, x, h):
    """Формула (4.1): (f(x+h) - f(x)) / h — правая разность, O(h)"""
    return (f(x + h) - f(x)) / h


def backward_diff(f, x, h):
    """Формула (4.2): (f(x) - f(x-h)) / h — левая разность, O(h)"""
    return (f(x) - f(x - h)) / h


def central_diff(f, x, h):
    """Формула (4.3): (f(x+h) - f(x-h)) / (2h) — центральная, O(h²)"""
    return (f(x + h) - f(x - h)) / (F32(2) * h)


def analyze_derivative_accuracy():
    """Пункт 1: исследование погрешности в float32."""
    print("=" * 100)
    print("ПУНКТ 1: ИССЛЕДОВАНИЕ ПОГРЕШНОСТИ ЧИСЛЕННОГО ДИФФЕРЕНЦИРОВАНИЯ")
    print("=" * 100)
    print(f"Вариант: {VARIANT}")
    print(f"Функция: f(x) = {VARIANT}·x²")
    print(f"Точная производная: f'(x) = {2 * VARIANT}·x")
    print(f"Тип данных: float32 (32 бит)")
    print(f"Машинная точность float32: eps = {np.finfo(F32).eps:.3e}")
    print()

    points = [F32(1e-2), F32(1e-1), F32(1), F32(10), F32(1e2)]
    h_values = [F32(10**(-k)) for k in range(1, 9)]  # 10^-1 .. 10^-8

    all_results = {}

    for pt in points:
        exact = f_deriv_exact(pt)
        print(f"\n--- Точка x = {float(pt):.0e},  f'(x) = {exact:.6e} ---")
        print(f"  {'h':>10s}   {'Левая(4.2)':>14s}  {'|err|':>10s}   "
              f"{'Центр.(4.3)':>14s}  {'|err|':>10s}   "
              f"{'Правая(4.1)':>14s}  {'|err|':>10s}")
        print("  " + "-" * 98)

        errs_l, errs_c, errs_r = [], [], []
        best = {'err': float('inf'), 'h': None, 'method': ''}

        for h in h_values:
            al = float(backward_diff(f_x, pt, h))
            ac = float(central_diff(f_x, pt, h))
            ar = float(forward_diff(f_x, pt, h))

            el = abs(al - exact)
            ec = abs(ac - exact)
            er = abs(ar - exact)

            errs_l.append(el)
            errs_c.append(ec)
            errs_r.append(er)

            print(f"  {float(h):10.1e}   {al:14.6e}  {el:10.2e}   "
                  f"{ac:14.6e}  {ec:10.2e}   "
                  f"{ar:14.6e}  {er:10.2e}")

            min_err = min(el, ec, er)
            if min_err < best['err']:
                best['err'] = min_err
                best['h'] = float(h)
                if min_err == el:
                    best['method'] = 'левая'
                elif min_err == ec:
                    best['method'] = 'центральная'
                else:
                    best['method'] = 'правая'

        print(f"\n  >>> Оптимальный шаг: h = {best['h']:.1e}, "
              f"метод: {best['method']}, ошибка = {best['err']:.2e}")

        all_results[float(pt)] = {
            'h': [float(hv) for hv in h_values],
            'err_l': errs_l, 'err_c': errs_c, 'err_r': errs_r,
        }

    # --- Графики погрешности ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, pt in enumerate([float(p) for p in points]):
        ax = axes[idx]
        res = all_results[pt]
        hh = res['h']

        def safe_log(arr):
            return [v if v > 0 else None for v in arr]

        for label, errs, style in [
            ('Левая (4.2)', res['err_l'], 'bo-'),
            ('Центр. (4.3)', res['err_c'], 'rs-'),
            ('Правая (4.1)', res['err_r'], 'g^--'),
        ]:
            vals = safe_log(errs)
            hh_f = [hh[i] for i in range(len(hh)) if vals[i] is not None]
            vv_f = [vals[i] for i in range(len(vals)) if vals[i] is not None]
            if vv_f:
                ax.loglog(hh_f, vv_f, style, label=label, markersize=5, linewidth=1.2)

        ax.set_xlabel('h')
        ax.set_ylabel('|Погрешность|')
        ax.set_title(f'x = {pt:.0e}')
        ax.legend(fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()

    axes[5].axis('off')
    eps32 = np.finfo(F32).eps
    axes[5].text(0.05, 0.5,
                 "Все вычисления в float32\n"
                 f"eps = {eps32:.2e}\n"
                 f"f(x) = {VARIANT}·x²\n"
                 f"f''(x) = {2*VARIANT}, f'''(x) = 0\n\n"
                 "Левая/правая: O(h), R~(h/2)·f''\n"
                 "Центральная: O(h²), R~(h²/6)·f'''\n"
                 "Для квадратичной f''≡0 ⟹\n"
                 "центральная даёт точный результат\n"
                 "до тех пор, пока не доминирует\n"
                 "погрешность округления float32.",
                 fontsize=9, verticalalignment='center',
                 transform=axes[5].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.suptitle('Пункт 1: Зависимость погрешности от шага h (float32)', fontsize=14)
    plt.tight_layout()
    plt.savefig('punkt1_error_vs_h.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nГрафик сохранён: punkt1_error_vs_h.png")

    eps32 = float(np.finfo(F32).eps)
    M2 = 2.0 * VARIANT  # f''(x) = 20
    print("\n" + "=" * 100)
    print("ТЕОРЕТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ ПУНКТА 1")
    print("=" * 100)
    print(f"""
    Тип данных: float32, машинная точность ε ≈ {eps32:.2e}
    Функция f(x) = {VARIANT}·x², f'(x) = {2*VARIANT}·x, f''(x) = {M2:.0f}, f'''(x) = 0.

    1. ЛЕВАЯ / ПРАВАЯ РАЗНОСТЬ (формулы 4.1, 4.2):
       Погрешность метода: R₁ = (h/2)·f''(ξ) = {M2/2:.0f}·h.
       Погрешность вычислительная: R₂ ≈ 2E/h, E ~ ε·|f(x)|.
       Общая: g(h) = {M2/2:.0f}·h + 2E/h.
       Оптимальный шаг: h_opt = sqrt(4E/{M2:.0f}) = 2·sqrt(ε·|f(x)|/{M2:.0f}).
       Для x=1: h_opt ≈ 2·sqrt({eps32:.2e}·{VARIANT}/{M2:.0f}) ≈ {2*np.sqrt(eps32*VARIANT/M2):.2e}.

    2. ЦЕНТРАЛЬНАЯ РАЗНОСТЬ (формула 4.3):
       Погрешность метода: R₁ = (h²/6)·f'''(ξ).
       Для f(x) = {VARIANT}·x²: f'''(x) ≡ 0, поэтому R₁ = 0.
       Центральная разность вычисляет производную ТОЧНО для квадратичной
       функции (ошибка метода нулевая). Остаётся лишь погрешность округления.
       При больших h ошибка центральной разности ≈ 0 (машинная точность),
       при малых h < {np.sqrt(eps32):.1e} начинается потеря значащих цифр.

    3. ЗАВИСИМОСТЬ ОТ ТОЧКИ x:
       Абсолютная погрешность округления пропорциональна |f(x)|,
       поэтому для больших x (x=100, f(100)={VARIANT*100**2:.0f}) потеря точности
       наступает при бо́льших h, чем для малых x.
    """)

    return all_results


# ======================== ПУНКТ 2 ========================
# "Для функции, заданной таблицей значений (таблица 3.2, вариант 10),
#  выполнить «вручную» расчёты коэффициентов кубического сплайна
#  по формулам (3.3), (3.7), (3.9)–(3.11)."
# ==========================================================

x_data = np.array([2, 3, 4, 6, 7], dtype=float)
y_data = np.array([-1, -6, 7, 8, 2], dtype=float)


def solve_tridiagonal(a_low, b_main, c_up, d_rhs):
    """Метод прогонки (Thomas algorithm) для трёхдиагональной системы."""
    n = len(d_rhs)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)

    cp[0] = c_up[0] / b_main[0]
    dp[0] = d_rhs[0] / b_main[0]

    for i in range(1, n):
        m = b_main[i] - a_low[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = c_up[i] / m
        dp[i] = (d_rhs[i] - a_low[i - 1] * dp[i - 1]) / m

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def calculate_cubic_spline(x, y):
    """
    Естественный кубический сплайн: S''(x₀)=0, S''(xₙ)=0.
    S_i(t) = a_i + b_i·t + c_i·t² + d_i·t³,  t = x − x_i.
    Возвращает a, b, c, d, h, c_all (включая c₀ и cₙ).
    """
    n = len(x) - 1
    h = np.diff(x)

    a = y[:-1].copy()  # (3.3): a_i = f(x_{i-1})

    # Система (3.11) для c₂..cₙ, c₁=0 (3.7), cₙ₊₁=0
    A_low = np.zeros(n - 2)
    A_main = np.zeros(n - 1)
    A_up = np.zeros(n - 2)
    rhs = np.zeros(n - 1)

    for i in range(1, n):
        row = i - 1
        A_main[row] = 2.0 * (h[i - 1] + h[i])
        if row > 0:
            A_low[row - 1] = h[i - 1]
        if row < n - 2:
            A_up[row] = h[i]
        rhs[row] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c_inner = solve_tridiagonal(A_low, A_main, A_up, rhs)

    c_all = np.zeros(n + 1)
    c_all[0] = 0.0       # (3.7)
    c_all[1:n] = c_inner
    c_all[n] = 0.0        # (3.12)

    d = np.zeros(n)
    b = np.zeros(n)
    c = c_all[:n]

    for i in range(n):
        d[i] = (c_all[i + 1] - c_all[i]) / (3.0 * h[i])      # (3.9)
        b[i] = (y[i + 1] - y[i]) / h[i] \
               - (c_all[i + 1] + 2.0 * c_all[i]) * h[i] / 3.0  # (3.10)

    return a, b, c, d, h, c_all


def spline_eval(xv, x_nodes, a, b, c, d):
    """Значение сплайна S(xv)."""
    n = len(a)
    for i in range(n - 1, -1, -1):
        if xv >= x_nodes[i]:
            t = xv - x_nodes[i]
            return a[i] + b[i] * t + c[i] * t**2 + d[i] * t**3
    t = xv - x_nodes[0]
    return a[0] + b[0] * t + c[0] * t**2 + d[0] * t**3


def spline_deriv2_analytical(xv, x_nodes, c, d):
    """Вторая производная S''(x) = 2c_i + 6d_i·t (аналитическая)."""
    n = len(c)
    for i in range(n - 1, -1, -1):
        if xv >= x_nodes[i]:
            t = xv - x_nodes[i]
            return 2.0 * c[i] + 6.0 * d[i] * t
    t = xv - x_nodes[0]
    return 2.0 * c[0] + 6.0 * d[0] * t


def spline_deriv1_central(xv, x_nodes, a, b, c, d, delta=1e-4):
    """
    Первая производная по формуле (4.3):
    S'(x) ≈ (S(x+δ) - S(x-δ)) / (2δ)
    """
    sp = spline_eval(xv + delta, x_nodes, a, b, c, d)
    sm = spline_eval(xv - delta, x_nodes, a, b, c, d)
    return (sp - sm) / (2.0 * delta)


def manual_calculation_demo():
    """Пункт 2: «ручной» расчёт коэффициентов с выводом системы уравнений."""
    print("\n" + "=" * 100)
    print("ПУНКТ 2: РАСЧЁТ КОЭФФИЦИЕНТОВ КУБИЧЕСКОГО СПЛАЙНА (Вариант 10)")
    print("=" * 100)

    print("\nИсходные данные (таблица 3.2, вариант 10):")
    print(f"  x    = {x_data.tolist()}")
    print(f"  f(x) = {y_data.tolist()}")

    n = len(x_data) - 1
    h = np.diff(x_data)

    print(f"\n  Количество интервалов n = {n}")
    print("\n  Шаги h_i = x_{{i+1}} − x_i:")
    for i in range(n):
        print(f"    h_{i+1} = x_{i+1} − x_{i} = {x_data[i+1]:.0f} − {x_data[i]:.0f} = {h[i]:.0f}")

    print("\n--- Система уравнений (3.11) для c_i ---")
    print("  Граничные условия (3.7) и (3.12): c_1 = 0, c_{n+1} = 0.")
    print("  Уравнения для внутренних узлов (i = 2, ..., n):")

    for i in range(1, n):
        rhs_val = 3.0 * ((y_data[i + 1] - y_data[i]) / h[i] -
                          (y_data[i] - y_data[i - 1]) / h[i - 1])
        terms = []
        if i > 1:
            terms.append(f"{h[i-1]:.0f}·c_{i}")
        terms.append(f"{2*(h[i-1]+h[i]):.0f}·c_{i+1}")
        if i < n - 1:
            terms.append(f"{h[i]:.0f}·c_{i+2}")
        print(f"    i={i+1}: {' + '.join(terms)} = {rhs_val:.4f}")

    print("\n  Подставляя c_1 = 0 и c_5 = 0, получаем систему 3×3:")

    A_mat = np.zeros((n - 1, n - 1))
    b_vec = np.zeros(n - 1)

    for i in range(1, n):
        row = i - 1
        if i - 1 >= 1:
            A_mat[row, row - 1] = h[i - 1]
        A_mat[row, row] = 2.0 * (h[i - 1] + h[i])
        if i + 1 <= n - 1:
            A_mat[row, row + 1] = h[i]
        b_vec[row] = 3.0 * ((y_data[i + 1] - y_data[i]) / h[i] -
                             (y_data[i] - y_data[i - 1]) / h[i - 1])

    print("\n  Матрица системы:")
    for row in range(n - 1):
        coeffs = "  |"
        for col in range(n - 1):
            coeffs += f" {A_mat[row, col]:6.1f}"
        coeffs += f" | · c_{row+2}   =  {b_vec[row]:10.4f}"
        print(coeffs)

    c_inner = np.linalg.solve(A_mat, b_vec)
    print(f"\n  Решение: c_2 = {c_inner[0]:.6f}, c_3 = {c_inner[1]:.6f}, "
          f"c_4 = {c_inner[2]:.6f}")

    a, b, c, d, hh, c_all = calculate_cubic_spline(x_data, y_data)

    print("\n--- Коэффициенты сплайна ---")
    print(f"\n  a_i = f(x_{{i-1}}) по формуле (3.3):")
    for i in range(n):
        print(f"    a_{i+1} = {a[i]:.6f}")

    print(f"\n  c_i (c_1={c_all[0]:.1f}, c_5={c_all[n]:.1f}):")
    for i in range(n):
        print(f"    c_{i+1} = {c[i]:.6f}")

    print(f"\n  d_i по формуле (3.9): d_i = (c_{{i+1}} − c_i) / (3·h_i)")
    for i in range(n):
        print(f"    d_{i+1} = ({c_all[i+1]:.6f} − {c_all[i]:.6f}) / "
              f"(3·{hh[i]:.0f}) = {d[i]:.6f}")

    print(f"\n  b_i по формуле (3.10):")
    for i in range(n):
        print(f"    b_{i+1} = ({y_data[i+1]:.0f}−({y_data[i]:.0f}))/{hh[i]:.0f} − "
              f"({c_all[i+1]:.6f}+2·{c_all[i]:.6f})·{hh[i]:.0f}/3 = {b[i]:.6f}")

    print("\n--- ИТОГОВАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ ---")
    print(f"  {'Сплайн':<10} {'Интервал':<12} {'a_i':>12} {'b_i':>12} {'c_i':>12} {'d_i':>12}")
    print("  " + "-" * 72)
    for i in range(n):
        print(f"  S_{i+1}(x)   [{x_data[i]:.0f}; {x_data[i+1]:.0f}]      "
              f"{a[i]:12.6f} {b[i]:12.6f} {c[i]:12.6f} {d[i]:12.6f}")

    return a, b, c, d, hh, c_all


# ======================== ПУНКТ 3 ========================
# "Разработать программную реализацию построения кубического сплайна.
#  Рассчитать и построить графики первой и второй производной.
#  Для первой производной использовать выражение (4.3)."
# ==========================================================


def plot_spline_and_derivatives(x_nodes, y_nodes, a, b, c, d, h):
    """Графики сплайна, первой производной (4.3) и второй производной."""
    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 500)
    delta = 1e-3  # шаг для центральной разности (4.3)

    y_spline = np.array([spline_eval(xi, x_nodes, a, b, c, d) for xi in x_fine])
    y_d1 = np.array([spline_deriv1_central(xi, x_nodes, a, b, c, d, delta)
                      for xi in x_fine])
    y_d2 = np.array([spline_deriv2_analytical(xi, x_nodes, c, d)
                      for xi in x_fine])

    d1_at_nodes = np.array([spline_deriv1_central(xi, x_nodes, a, b, c, d, delta)
                             for xi in x_nodes])
    d2_at_nodes = np.array([spline_deriv2_analytical(xi, x_nodes, c, d)
                             for xi in x_nodes])

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    axes[0].plot(x_fine, y_spline, 'b-', linewidth=2, label='Кубический сплайн S(x)')
    axes[0].scatter(x_nodes, y_nodes, color='red', s=100, zorder=5,
                    label='Точки таблицы', marker='o', edgecolors='black', linewidths=1.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('S(x)')
    axes[0].set_title('Кубический сплайн (интерполяция)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_fine, y_d1, 'g-', linewidth=2,
                 label=f"S'(x) по формуле (4.3), δ={delta:.0e}")
    axes[1].scatter(x_nodes, d1_at_nodes, color='red', s=80, zorder=5,
                    label='Значения в узлах', marker='s',
                    edgecolors='black', linewidths=1.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel("S'(x)")
    axes[1].set_title('Первая производная (центральная разность, формула 4.3)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_fine, y_d2, 'r-', linewidth=2, label="S''(x) (аналитическая)")
    axes[2].scatter(x_nodes, d2_at_nodes, color='blue', s=80, zorder=5,
                    label='Значения в узлах', marker='^',
                    edgecolors='black', linewidths=1.5)
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

    print(f"\n--- ПРОВЕРКА НЕПРЕРЫВНОСТИ ПРОИЗВОДНЫХ ВО ВНУТРЕННИХ УЗЛАХ ---")
    print(f"  (первая производная вычислена по формуле (4.3) с δ={delta:.0e})")
    for i in range(1, len(x_nodes) - 1):
        eps = 1e-6
        d1_l = spline_deriv1_central(x_nodes[i] - eps, x_nodes, a, b, c, d, delta)
        d1_r = spline_deriv1_central(x_nodes[i] + eps, x_nodes, a, b, c, d, delta)
        d2_l = spline_deriv2_analytical(x_nodes[i] - eps, x_nodes, c, d)
        d2_r = spline_deriv2_analytical(x_nodes[i] + eps, x_nodes, c, d)

        print(f"  Узел x = {x_nodes[i]:.0f}:")
        print(f"    S'(x−ε)  = {d1_l:.8f},  S'(x+ε)  = {d1_r:.8f},  "
              f"|разн.| = {abs(d1_l - d1_r):.2e}")
        print(f"    S''(x−ε) = {d2_l:.8f},  S''(x+ε) = {d2_r:.8f},  "
              f"|разн.| = {abs(d2_l - d2_r):.2e}")


def compare_with_scipy(x_nodes, y_nodes, a, b, c, d):
    """Сравнение с scipy.interpolate.CubicSpline (bc_type='natural')."""
    cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')

    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 300)
    y_scipy = cs(x_fine)

    y_ours = np.array([spline_eval(xi, x_nodes, a, b, c, d) for xi in x_fine])

    err = np.max(np.abs(y_scipy - y_ours))

    print(f"\n--- СРАВНЕНИЕ С SCIPY ---")
    print(f"  max|S_наш(x) − S_scipy(x)| = {err:.2e}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x_fine, y_scipy, 'b-', label='scipy CubicSpline', linewidth=2)
    ax.plot(x_fine, y_ours, 'r--', label='Наша реализация', linewidth=2)
    ax.scatter(x_nodes, y_nodes, color='black', s=60, zorder=5, label='Узлы')
    ax.set_title('Сравнение сплайнов: наша реализация vs scipy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_with_scipy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  График сохранён: comparison_with_scipy.png")


def numerical_derivatives_at_nodes(x_nodes, y_nodes, a, b, c, d, h):
    """Численные производные в узлах таблицы (формулы 4.1-4.3, неравномерный шаг)."""
    print("\n--- ЧИСЛЕННЫЕ ПРОИЗВОДНЫЕ В УЗЛАХ ТАБЛИЦЫ ---")
    print("  (сравнение разностных формул с производной сплайна)")
    n = len(x_nodes)

    delta = 1e-3
    col_sd = "S'(x_i)(4.3)"
    print(f"\n  {'i':<4} {'x_i':<6} {'f(x_i)':<8} {'Числ.произв.':<16} "
          f"{col_sd:<16} {'|Разн.|':<12} {'Формула'}")
    print("  " + "-" * 85)

    for i in range(n):
        s_d = spline_deriv1_central(x_nodes[i], x_nodes, a, b, c, d, delta)

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

        diff = abs(num_d - s_d)
        print(f"  {i:<4} {x_nodes[i]:<6.1f} {y_nodes[i]:<8.1f} "
              f"{num_d:<16.6f} {s_d:<16.6f} {diff:<12.2e} {method}")


# ======================== MAIN ========================


def main():
    print("╔" + "═" * 98 + "╗")
    print("║  ЛАБОРАТОРНАЯ РАБОТА №4: ЧИСЛЕННОЕ ДИФФЕРЕНЦИРОВАНИЕ".ljust(99) + "║")
    print("║  Вариант 10".ljust(99) + "║")
    print("║  Ландовский В.В. «Численные методы», НГТУ, 2022".ljust(99) + "║")
    print("╚" + "═" * 98 + "╝")

    # --- Пункт 1 ---
    analyze_derivative_accuracy()

    # --- Пункт 2 ---
    a, b, c, d, h, c_all = manual_calculation_demo()

    # --- Пункт 3 ---
    print("\n" + "=" * 100)
    print("ПУНКТ 3: ПРОГРАММНАЯ РЕАЛИЗАЦИЯ — ГРАФИКИ И ПРОВЕРКИ")
    print("=" * 100)

    plot_spline_and_derivatives(x_data, y_data, a, b, c, d, h)
    compare_with_scipy(x_data, y_data, a, b, c, d)
    numerical_derivatives_at_nodes(x_data, y_data, a, b, c, d, h)

    # --- Проверка свойств ---
    print("\n" + "=" * 100)
    print("ПРОВЕРКА СВОЙСТВ СПЛАЙНА")
    print("=" * 100)

    print("\n  1. Интерполяция S(x_i) = f(x_i):")
    for i, xi in enumerate(x_data):
        si = spline_eval(xi, x_data, a, b, c, d)
        print(f"     S({xi:.0f}) = {si:.10f},  f(x_{i}) = {y_data[i]:.1f},  "
              f"|разн.| = {abs(si - y_data[i]):.2e}")

    print("\n  2. Естественные граничные условия S''(x₀)=0, S''(xₙ)=0:")
    d2_left = spline_deriv2_analytical(x_data[0], x_data, c, d)
    d2_right = spline_deriv2_analytical(x_data[-1], x_data, c, d)
    print(f"     S''({x_data[0]:.0f}) = {d2_left:.10f}  (должно быть 0)")
    print(f"     S''({x_data[-1]:.0f}) = {d2_right:.10f}  (должно быть 0)")

    # --- Выводы ---
    print("\n" + "=" * 100)
    print("ВЫВОДЫ ПО РАБОТЕ")
    print("=" * 100)
    print(f"""
    1. Пункт 1 (исследование погрешности, float32):
       - Вычисления выполнены в 32-битном представлении (eps ≈ 1.19e-07).
       - Шаги: 10⁻¹, 10⁻², ..., 10⁻⁸ (как задано в методичке).
       - Центральная разность (4.3) для f(x)={VARIANT}·x² даёт точный результат
         при умеренных h, т.к. f'''(x) ≡ 0.
       - Левая/правая разности (4.1, 4.2) имеют погрешность O(h),
         оптимальный шаг h_opt ≈ sqrt(ε) ≈ 3·10⁻⁴.
       - При h < 10⁻⁴ доминирует погрешность округления float32.

    2. Пункт 2 (расчёт коэффициентов сплайна):
       - Построен естественный кубический сплайн по 5 узлам {x_data.tolist()}.
       - Коэффициенты рассчитаны по формулам (3.3), (3.7), (3.9)–(3.11).
       - Система 3×3 решена методом прогонки.
       - Результат совпадает с scipy.interpolate.CubicSpline.

    3. Пункт 3 (графики и проверки):
       - Сплайн точно проходит через все узлы таблицы.
       - Первая производная вычислена по формуле (4.3) — центральная разность.
       - Непрерывность S' и S'' подтверждена во внутренних узлах.
       - Граничные условия S''(x₀)=S''(xₙ)=0 выполнены.
    """)


if __name__ == "__main__":
    main()
