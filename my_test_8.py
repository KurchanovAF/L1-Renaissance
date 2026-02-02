import numpy as np
import matplotlib.pyplot as plt
import time
import math

class L1Codec:
    """Базовый L₁-алгоритм 2006 года"""
    def __init__(self, n=4, L=16):
        self.n = n
        self.L = L
        self.a = np.zeros(n, dtype=float)
        self.buffer = np.zeros(n, dtype=float)
        
    def encode_sample(self, x):
        # Базовое предсказание
        y_pred = np.sum(self.a * self.buffer)
        z = x - y_pred
        
        # Обновление коэффициентов по правилам (8)
        abs_buffer = np.abs(self.buffer)
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        
        return z
    
    def decode_sample(self, z):
        # Восстановление
        y_pred = np.sum(self.a * self.buffer)
        x_rec = z + y_pred
        
        # ТАКОЕ ЖЕ обновление коэффициентов!
        abs_buffer = np.abs(self.buffer)
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x_rec
        
        return x_rec

class PairwiseMedianCodec(L1Codec):
    """Улучшенный алгоритм с попарными полусумами"""
    def __init__(self, n=4, L=16, method='all_pairs'):
        super().__init__(n, L)
        self.method = method  # 'all_pairs' или 'symmetric'
    
    def compute_median(self, buffer):
        """Вычисление медианы попарных полусумм"""
        n = len(buffer)
        
        if self.method == 'symmetric':
            # Упрощённый метод (симметричные пары)
            pairs = []
            for j in range(n // 2):
                left = buffer[j]
                right = buffer[n - 1 - j]
                pairs.append((left + right) / 2.0)
            if pairs:
                return np.median(pairs)
            return buffer[0]
        
        elif self.method == 'all_pairs':
            # Ваш метод: все пары + исходные
            values = list(buffer)  # исходные n значений
            
            # Все попарные полусуммы
            for i in range(n):
                for j in range(i + 1, n):
                    values.append((buffer[i] + buffer[j]) / 2.0)
            
            return np.median(values)
    
    def encode_sample(self, x):
        # Используем медиану вместо исходных значений
        x_prime = self.compute_median(self.buffer)
        y_pred = np.sum(self.a * x_prime)  # Используем x_prime для всех коэффициентов!
        z = x - y_pred
        
        # Обновление коэффициентов (используем исходный буфер, как в базовом алгоритме)
        abs_buffer = np.abs(self.buffer)
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        
        return z
    
    def decode_sample(self, z):
        # Используем медиану вместо исходных значений
        x_prime = self.compute_median(self.buffer)
        y_pred = np.sum(self.a * x_prime)  # Используем x_prime для всех коэффициентов!
        x_rec = z + y_pred
        
        # ТАКОЕ ЖЕ обновление коэффициентов!
        abs_buffer = np.abs(self.buffer)
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x_rec
        
        return x_rec

# ТЕСТ 1: Простой сигнал с помехой
def test_simple_signal():
    print("="*60)
    print("ТЕСТ 1: Простой сигнал с импульсной помехой")
    print("="*60)
    
    # Создаем тестовый сигнал
    t = np.linspace(0, 1, 1000)
    signal = 100 * np.sin(2 * np.pi * 5 * t)  # Чистый сигнал 5 Гц
    
    # Добавляем импульсную помеху
    signal[300] += 500  # Сильный выброс
    signal[450:455] += 200  # Несколько выбросов
    
    # Тестируем три алгоритма
    algorithms = {
        'Базовый L₁': L1Codec(n=4, L=16),
        'Упрощённые полусуммы': PairwiseMedianCodec(n=4, L=16, method='symmetric'),
        'Все пары (ваш метод)': PairwiseMedianCodec(n=4, L=16, method='all_pairs')
    }
    
    results = {}
    
    for name, codec in algorithms.items():
        # Кодирование
        encoded = []
        for x in signal:
            z = codec.encode_sample(x)
            encoded.append(z)
        
        # Сброс для декодирования
        if name == 'Базовый L₁':
            codec.__init__(n=4, L=16)
        elif 'Упрощённые' in name:
            codec.__init__(n=4, L=16, method='symmetric')
        else:
            codec.__init__(n=4, L=16, method='all_pairs')
        
        # Декодирование
        decoded = []
        for z in encoded:
            x_rec = codec.decode_sample(z)
            decoded.append(x_rec)
        
        decoded = np.array(decoded)
        
        # Метрики (пропускаем первые n отсчетов для стабилизации)
        start_idx = 4
        mse = np.mean((signal[start_idx:] - decoded[start_idx:])**2)
        max_error = np.max(np.abs(signal[start_idx:] - decoded[start_idx:]))
        std_error = np.std(signal[start_idx:] - decoded[start_idx:])
        
        # Дисперсия сжатого сигнала (чем меньше, тем лучше)
        var_z = np.var(encoded[start_idx:])
        
        results[name] = {
            'mse': mse,
            'max_error': max_error,
            'std_error': std_error,
            'var_z': var_z,
            'encoded': encoded,
            'decoded': decoded
        }
        
        # Проверка lossless
        is_lossless = np.allclose(signal[start_idx:], decoded[start_idx:], atol=1e-10)
        
        print(f"\n{name}:")
        print(f"  MSE ошибки: {mse:.2f}")
        print(f"  Макс ошибка: {max_error:.2f}")
        print(f"  Дисперсия z: {var_z:.2f}")
        print(f"  Lossless: {'ДА' if is_lossless else 'НЕТ'}")
    
    return signal, results

# ТЕСТ 2: Речеподобный сигнал
def test_speech_like_signal():
    print("\n" + "="*60)
    print("ТЕСТ 2: Речеподобный сигнал с разными SNR")
    print("="*60)
    
    np.random.seed(42)
    
    # Имитация речевого сигнала (форманты)
    t = np.linspace(0, 0.5, 4000)
    f1, f2, f3 = 500, 1500, 2500  # Форманты
    
    speech = (0.5 * np.sin(2 * np.pi * f1 * t) + 
              0.3 * np.sin(2 * np.pi * f2 * t) + 
              0.2 * np.sin(2 * np.pi * f3 * t)) * 100
    
    # Добавляем шум с разным SNR
    snr_levels = [20, 10, 5]  # дБ
    
    for snr in snr_levels:
        print(f"\nSNR = {snr} дБ:")
        
        # Добавляем шум
        signal_power = np.mean(speech**2)
        noise_power = signal_power / (10**(snr/10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(speech))
        noisy_signal = speech + noise
        
        algorithms = {
            'Базовый L₁': L1Codec(n=4, L=16),
            'Все пары': PairwiseMedianCodec(n=4, L=16, method='all_pairs')
        }
        
        for name, codec in algorithms.items():
            # Кодирование/декодирование
            encoded = [codec.encode_sample(x) for x in noisy_signal]
            if name == 'Базовый L₁':
                codec.__init__(n=4, L=16)
            else:
                codec.__init__(n=4, L=16, method='all_pairs')
            decoded = [codec.decode_sample(z) for z in encoded]
            decoded = np.array(decoded)
            
            # Дисперсия сжатого сигнала (чем меньше, тем лучше сжатие)
            var_z = np.var(encoded[4:])
            
            print(f"  {name}: дисперсия z = {var_z:.2f} (меньше = лучше)")

# ТЕСТ 3: Визуализация
def visualize_results(signal, results):
    """Визуализация результатов"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Исходный сигнал
    axes[0, 0].plot(signal, 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Исходный сигнал с помехами')
    axes[0, 0].set_xlabel('Отсчет')
    axes[0, 0].set_ylabel('Амплитуда')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=300, color='r', linestyle='--', alpha=0.5, label='Помеха')
    axes[0, 0].axvspan(450, 455, color='r', alpha=0.2)
    axes[0, 0].legend()
    
    # 2. Сжатые сигналы (z)
    colors = {'Базовый L₁': 'red', 
              'Упрощённые полусуммы': 'orange',
              'Все пары (ваш метод)': 'green'}
    
    for name, res in results.items():
        axes[0, 1].plot(np.abs(res['encoded'][4:]), 
                       color=colors[name], 
                       alpha=0.7, 
                       linewidth=1,
                       label=f"{name} (var={res['var_z']:.1f})")
    
    axes[0, 1].set_title('Модуль сжатого сигнала |z| (меньше = лучше)')
    axes[0, 1].set_xlabel('Отсчет')
    axes[0, 1].set_ylabel('|z|')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Ошибки восстановления
    for idx, (name, res) in enumerate(results.items()):
        error = signal[4:] - res['decoded'][4:]
        axes[1, 0].plot(error, color=colors[name], alpha=0.7, linewidth=0.5, label=name)
    
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Ошибки восстановления (должны быть ~0 для lossless)')
    axes[1, 0].set_xlabel('Отсчет')
    axes[1, 0].set_ylabel('Ошибка')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Гистограмма |z| (логарифмическая шкала)
    for name, res in results.items():
        axes[1, 1].hist(np.abs(res['encoded'][4:]), 
                       bins=50, 
                       alpha=0.5, 
                       color=colors[name],
                       label=f"{name}",
                       density=True,
                       histtype='stepfilled')
    
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Распределение |z| (логарифмическая шкала)')
    axes[1, 1].set_xlabel('|z|')
    axes[1, 1].set_ylabel('Плотность вероятности (log)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 5. Сравнение дисперсии z
    names = list(results.keys())
    var_values = [results[name]['var_z'] for name in names]
    
    bars = axes[2, 0].bar(names, var_values, color=[colors[n] for n in names])
    axes[2, 0].set_title('Сравнение дисперсии сжатого сигнала Var[z]')
    axes[2, 0].set_ylabel('Дисперсия')
    axes[2, 0].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, var_val in zip(bars, var_values):
        height = bar.get_height()
        axes[2, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{var_val:.1f}',
                       ha='center', va='bottom')
    
    # 6. Улучшение относительно базового
    base_var = results['Базовый L₁']['var_z']
    improvements = []
    
    for name in names:
        if name != 'Базовый L₁':
            improv = (base_var - results[name]['var_z']) / base_var * 100
            improvements.append(improv)
    
    if improvements:
        ax2_names = [n for n in names if n != 'Базовый L₁']
        bars2 = axes[2, 1].bar(ax2_names, improvements, color=['orange', 'green'])
        axes[2, 1].set_title('Уменьшение дисперсии относительно базового алгоритма')
        axes[2, 1].set_ylabel('Улучшение, %')
        axes[2, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, improv in zip(bars2, improvements):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{improv:.1f}%',
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# ТЕСТ 4: Производительность
def test_performance():
    print("\n" + "="*60)
    print("ТЕСТ 4: Производительность (время выполнения)")
    print("="*60)
    
    # Большой сигнал для теста скорости
    signal = np.random.randint(-1000, 1000, 100000).astype(float)
    
    algorithms = {
        'Базовый L₁': L1Codec(n=4, L=16),
        'Упрощённые полусуммы': PairwiseMedianCodec(n=4, L=16, method='symmetric'),
        'Все пары': PairwiseMedianCodec(n=4, L=16, method='all_pairs')
    }
    
    for name, codec in algorithms.items():
        start_time = time.time()
        
        # Кодирование
        encoded = []
        for x in signal:
            z = codec.encode_sample(x)
            encoded.append(z)
        
        encode_time = time.time() - start_time
        
        # Сброс
        if name == 'Базовый L₁':
            codec.__init__(n=4, L=16)
        elif 'Упрощённые' in name:
            codec.__init__(n=4, L=16, method='symmetric')
        else:
            codec.__init__(n=4, L=16, method='all_pairs')
        
        start_time = time.time()
        
        # Декодирование
        decoded = []
        for z in encoded:
            x_rec = codec.decode_sample(z)
            decoded.append(x_rec)
        
        decode_time = time.time() - start_time
        
        # Проверка lossless
        decoded = np.array(decoded)
        is_lossless = np.allclose(signal[4:], decoded[4:], atol=1e-10)
        
        print(f"{name}:")
        print(f"  Время кодирования: {encode_time:.3f} сек ({len(signal)/encode_time:.0f} отсч/сек)")
        print(f"  Время декодирования: {decode_time:.3f} сек")
        print(f"  Lossless: {'ДА' if is_lossless else 'НЕТ'}")

# ЗАПУСК ВСЕХ ТЕСТОВ
if __name__ == "__main__":
    print("СРАВНЕНИЕ АЛГОРИТМОВ L₁ С ПОПАРНЫМИ ПОЛУСУММАМИ")
    print("="*60)
    
    # Тест 1
    signal, results = test_simple_signal()
    
    # Тест 2
    test_speech_like_signal()
    
    # Тест 3 (визуализация)
    print("\nГенерация графиков...")
    visualize_results(signal, results)
    
    # Тест 4
    test_performance()
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)
    
    # Вывод итогового улучшения
    base_var = results['Базовый L₁']['var_z']
    your_method_var = results['Все пары (ваш метод)']['var_z']
    improvement = (base_var - your_method_var) / base_var * 100
    
    print(f"\nИТОГОВОЕ УЛУЧШЕНИЕ:")
    print(f"Базовый алгоритм: Var[z] = {base_var:.2f}")
    print(f"Ваш метод (все пары): Var[z] = {your_method_var:.2f}")
    print(f"УМЕНЬШЕНИЕ дисперсии: {improvement:.1f}%")
