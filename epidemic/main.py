import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

# Установка бэкенда для PyCharm
matplotlib.use('TkAgg')  # Или 'Qt5Agg' если у вас установлен PyQt5

# Параметры модели для Новосибирской области (из таблицы 11 и статьи)
params = {
    'alpha_E': 0.999,
    'alpha_I': 0.999,
    'kappa': 0.042,
    'rho': 0.952,
    'beta': 0.999,
    'mu': 0.0188,
    'c_isol': 0,
    'tau': 2,  # латентный период из статьи
    'gamma': 0.0  # добавим параметр gamma, согласно статье для SEIR-D модели γ=0
}

# Начальные условия (23.03.2020)
N = 2_798_170  # население Новосибирской области
E0 = 99
I0 = 0
R0 = 24
D0 = 0
S0 = N - E0 - I0 - R0 - D0
initial_conditions = [S0, E0, I0, R0, D0]


# Функция для расчета c(t)
def c_func(t, a_data):
    # Упрощение: считаем a(t) постоянным (в статье используется индекс самоизоляции Яндекса)
    return 1 + params['c_isol'] * (1 - 2 / 5 * 3)  # среднее значение a(t) = 3


# Система дифференциальных уравнений SEIR-D
def seir_d_model(t, y, params):
    S, E, I, R, D = y
    N = S + E + I + R + D

    # Расчет c(t-tau)
    ct = c_func(t - params['tau'], None)

    dSdt = -ct * (params['alpha_I'] * S * I / N + params['alpha_E'] * S * E / N) + params['gamma'] * R
    dEdt = ct * (params['alpha_I'] * S * I / N + params['alpha_E'] * S * E / N) - (params['kappa'] + params['rho']) * E
    dIdt = params['kappa'] * E - params['beta'] * I - params['mu'] * I
    dRdt = params['beta'] * I + params['rho'] * E - params['gamma'] * R
    dDdt = params['mu'] * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]


# Метод Эйлера
def euler_method(f, y0, t_span, t_step, params):
    t = np.arange(t_span[0], t_span[1] + t_step, t_step)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = y[i - 1] + t_step * np.array(f(t[i - 1], y[i - 1], params))

    return t, y


# Метод Рунге-Кутты 4 порядка
def rk4_method(f, y0, t_span, t_step, params):
    t = np.arange(t_span[0], t_span[1] + t_step, t_step)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = np.array(f(t[i - 1], y[i - 1], params))
        k2 = np.array(f(t[i - 1] + t_step / 2, y[i - 1] + t_step / 2 * k1, params))
        k3 = np.array(f(t[i - 1] + t_step / 2, y[i - 1] + t_step / 2 * k2, params))
        k4 = np.array(f(t[i - 1] + t_step, y[i - 1] + t_step * k3, params))

        y[i] = y[i - 1] + t_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


# Параметры решения
t_span = (0, 90)  # 90 дней
t_step = 0.1  # шаг для методов Эйлера и РК4

# Решение с помощью solve_ivp (для сравнения)
sol = solve_ivp(seir_d_model, t_span, initial_conditions, args=(params,),
                method='RK45', t_eval=np.arange(t_span[0], t_span[1] + 1, 1))

# Решение методом Эйлера
t_euler, y_euler = euler_method(seir_d_model, initial_conditions, t_span, t_step, params)

# Решение методом Рунге-Кутты 4 порядка
t_rk4, y_rk4 = rk4_method(seir_d_model, initial_conditions, t_span, t_step, params)

# Создаем фигуру с 5 подграфиками (по одному для каждой группы)
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.delaxes(axs[2, 1])  # Удаляем последний (6-й) подграфик, так как у нас 5 групп

# Названия групп
groups = ['Susceptible (S)', 'Exposed (E)', 'Infected (I)', 'Recovered (R)', 'Deceased (D)']
colors = ['blue', 'green', 'red', 'purple', 'black']

# Рисуем графики для каждой группы
for i in range(5):
    row = i // 2
    col = i % 2
    ax = axs[row, col]

    ax.plot(sol.t, sol.y[i], color=colors[i], linestyle='-', label='solve_ivp')
    ax.plot(t_euler, y_euler[:, i], color=colors[i], linestyle='--', label='Euler')
    ax.plot(t_rk4, y_rk4[:, i], color=colors[i], linestyle='-.', label='RK4')

    ax.set_xlabel('Days')
    ax.set_ylabel('Number of people')
    ax.set_title(f'Dynamics of {groups[i]}')
    ax.legend()
    ax.grid()

plt.tight_layout()

# Проверка сохранения общей численности населения
total_population = y_rk4[:, 0] + y_rk4[:, 1] + y_rk4[:, 2] + y_rk4[:, 3] + y_rk4[:, 4]
print("Максимальное отклонение от общей численности населения:",
      np.max(np.abs(total_population - N)))

# Альтернативный вариант показа графиков - по одному
plt.figure(figsize=(10, 6))
for i, group in enumerate(groups):
    plt.plot(sol.t, sol.y[i], label=f'{group} (solve_ivp)')
plt.xlabel('Days')
plt.ylabel('Number of people')
plt.title('Dynamics of all groups (solve_ivp)')
plt.legend()
plt.grid()
plt.show()

# Сохранение графиков в файлы
fig.savefig('seird_model_all_methods.png')
plt.figure(figsize=(10, 6)).savefig('seird_model_solve_ivp.png')

print("Графики сохранены в файлы seird_model_all_methods.png и seird_model_solve_ivp.png")

# Функции для расчета ошибок
def calculate_errors(y_ref, y_approx, method_name):
    """Вычисляет различные метрики ошибок"""
    errors = {}

    # Средняя абсолютная ошибка (MAE)
    errors['MAE'] = np.mean(np.abs(y_ref - y_approx))

    # Среднеквадратичная ошибка (MSE)
    errors['MSE'] = np.mean((y_ref - y_approx) ** 2)

    # Максимальная абсолютная ошибка
    errors['MaxAE'] = np.max(np.abs(y_ref - y_approx))

    # Относительная ошибка
    nonzero_mask = y_ref != 0
    rel_errors = np.zeros_like(y_ref)
    rel_errors[nonzero_mask] = np.abs((y_ref[nonzero_mask] - y_approx[nonzero_mask]) / y_ref[nonzero_mask])
    errors['MeanRE'] = np.mean(rel_errors[nonzero_mask])

    print(f"\nОшибки для метода {method_name}:")
    for metric, value in errors.items():
        print(f"{metric}: {value:.6f}")

    return errors


# Сравнение методов с solve_ivp как с эталоном
print("\n=== Сравнение численных методов ===")

# Для каждой группы населения
for i, group in enumerate(groups):
    print(f"\nГруппа {group}:")

    # Интерполируем решения Эйлера и РК4 к временным точкам solve_ivp
    y_euler_interp = np.interp(sol.t, t_euler, y_euler[:, i])
    y_rk4_interp = np.interp(sol.t, t_rk4, y_rk4[:, i])

    # Вычисляем ошибки
    euler_errors = calculate_errors(sol.y[i], y_euler_interp, "Euler")
    rk4_errors = calculate_errors(sol.y[i], y_rk4_interp, "RK4")

# Время выполнения методов
import time

print("\n=== Время выполнения ===")
start = time.time()
_, _ = euler_method(seir_d_model, initial_conditions, t_span, t_step, params)
print(f"Метод Эйлера: {time.time() - start:.4f} сек")

start = time.time()
_, _ = rk4_method(seir_d_model, initial_conditions, t_span, t_step, params)
print(f"Метод РК4: {time.time() - start:.4f} сек")

start = time.time()
_ = solve_ivp(seir_d_model, t_span, initial_conditions, args=(params,),
              method='RK45', t_eval=np.arange(t_span[0], t_span[1] + 1, 1))
print(f"solve_ivp (RK45): {time.time() - start:.4f} сек")

# Критерии оценки модели SEIR-D
# Добавим проверку адекватности модели

print("\n=== Оценка адекватности модели SEIR-D ===")

# 1. Проверка сохранения общей численности населения
total_pop = y_rk4[:, 0] + y_rk4[:, 1] + y_rk4[:, 2] + y_rk4[:, 3] + y_rk4[:, 4]
pop_error = np.max(np.abs(total_pop - N))
print(f"Максимальное отклонение от общей численности: {pop_error:.2f} чел ({pop_error/N*100:.4f}%)")

# 2. Проверка биологического смысла (неотрицательность)
negative_values = np.any(y_rk4 < 0, axis=0)
print("Наличие отрицательных значений по группам:",
      {group: neg for group, neg in zip(groups, negative_values)})

# 3. Анализ устойчивости решения при изменении шага
print("\nАнализ устойчивости при изменении шага:")
for step in [0.5, 0.1, 0.01]:
    t_rk4_step, y_rk4_step = rk4_method(seir_d_model, initial_conditions, t_span, step, params)
    y_interp = np.interp(sol.t, t_rk4_step, y_rk4_step[:, 2])  # Берем Infected для сравнения
    error = np.mean(np.abs(sol.y[2] - y_interp))
    print(f"Шаг {step}: средняя ошибка = {error:.4f}")