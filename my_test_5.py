# visualization_results.py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from core_2006 import L1AdaptiveFilter2006

class LMSFilter:
    def __init__(self, order=4, mu=0.01):
        self.order = order
        self.mu = mu
        self.coeffs = np.zeros(order)
    
    def process(self, signal):
        n = len(signal)
        compressed = np.zeros(n - self.order)
        coeff_history = []
        for i in range(self.order, n):
            x_past = signal[i-self.order:i][::-1]
            error = signal[i] - np.dot(self.coeffs, x_past)
            self.coeffs += self.mu * error * x_past
            compressed[i-self.order] = error
            coeff_history.append(self.coeffs.copy())
        return compressed, np.array(coeff_history)

# Создаем тестовый сигнал с выбросами
np.random.seed(42)
n_samples = 500
signal = np.random.randn(n_samples)
for i in range(4, n_samples):
    signal[i] += 0.8*signal[i-1] - 0.5*signal[i-2]

# Добавляем ВЫБРОСЫ в конкретных местах
outlier_positions = [100, 200, 300, 400]
for pos in outlier_positions:
    signal[pos] += 15.0  # Большой выброс

# Обработка L1
l1_filter = L1AdaptiveFilter2006(order=4, mu=0.03)
l1_compressed, l1_coeffs = l1_filter.process(signal)

# Обработка LMS
lms_filter = LMSFilter(order=4, mu=0.01)
lms_compressed, lms_coeffs = lms_filter.process(signal)

# Визуализация
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# 1. Исходный сигнал с выбросами
axes[0, 0].plot(signal, 'b-', alpha=0.7, linewidth=1)
axes[0, 0].scatter(outlier_positions, signal[outlier_positions], 
                  color='red', s=100, zorder=5, label='Выбросы')
axes[0, 0].set_xlabel('Отсчет')
axes[0, 0].set_ylabel('Амплитуда')
axes[0, 0].set_title('Исходный сигнал с выбросами')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Сжатые сигналы (L1 vs LMS)
axes[0, 1].plot(l1_compressed, 'g-', alpha=0.7, label='L1 сжатый', linewidth=1)
axes[0, 1].plot(lms_compressed, 'r-', alpha=0.3, label='LMS сжатый', linewidth=1)
axes[0, 1].set_xlabel('Отсчет')
axes[0, 1].set_ylabel('Ошибка предсказания')
axes[0, 1].set_title('Сжатые сигналы (ошибки предсказания)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Эволюция коэффициентов L1
for i in range(4):
    axes[1, 0].plot(l1_coeffs[:, i], label=f'L1 coeff {i}', alpha=0.8, linewidth=1.5)
axes[1, 0].set_xlabel('Итерация')
axes[1, 0].set_ylabel('Значение коэффициента')
axes[1, 0].set_title('Эволюция коэффициентов L1 алгоритма')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Эволюция коэффициентов LMS (КАРАСТРОФА!)
for i in range(4):
    axes[1, 1].plot(lms_coeffs[:, i], label=f'LMS coeff {i}', alpha=0.8, linewidth=1.5)
axes[1, 1].set_xlabel('Итерация')
axes[1, 1].set_ylabel('Значение коэффициента')
axes[1, 1].set_title('Эволюция коэффициентов LMS (разрушение при выбросах)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 5. Гистограммы ошибок
axes[2, 0].hist(l1_compressed, bins=50, alpha=0.7, color='green', 
               edgecolor='black', label='L1 ошибки')
axes[2, 0].set_xlabel('Ошибка предсказания')
axes[2, 0].set_ylabel('Частота')
axes[2, 0].set_title('Распределение ошибок L1 (стабильное)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].hist(lms_compressed, bins=50, alpha=0.7, color='red',
               edgecolor='black', label='LMS ошибки', range=(-50, 50))
axes[2, 1].set_xlabel('Ошибка предсказания')
axes[2, 1].set_ylabel('Частота')
axes[2, 1].set_title('Распределение ошибок LMS (катастрофическое)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.suptitle('L1 vs LMS: Катастрофическая неустойчивость L2-метрики к выбросам', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('l1_vs_lms_catastrophe.png', dpi=150, bbox_inches='tight')
plt.show()

# Статистика
print("="*70)
print("СТАТИСТИКА КАТАСТРОФЫ LMS:")
print("="*70)
print(f"Максимальная ошибка L1:  {np.max(np.abs(l1_compressed)):.2f}")
print(f"Максимальная ошибка LMS: {np.max(np.abs(lms_compressed)):.2f}")
print(f"Отношение (LMS/L1):      {np.max(np.abs(lms_compressed))/np.max(np.abs(l1_compressed)):.0f}X")
print()
print(f"Дисперсия L1 ошибок:     {np.var(l1_compressed):.4f}")
print(f"Дисперсия LMS ошибок:    {np.var(lms_compressed):.4f}")
print(f"Отношение дисперсий:     {np.var(lms_compressed)/np.var(l1_compressed):.0f}X")
