import numpy as np
import matplotlib.pyplot as plt
import time

class L1Codec:
    """Базовый L₁-алгоритм 2006 года"""
    def __init__(self, n=4, L=16):
        self.n = n  # порядок предсказания
        self.L = L  # порог обновления
        self.a = np.zeros(n, dtype=float)  # коэффициенты предсказания
        self.buffer = np.zeros(n, dtype=float)  # история отсчетов
        
    def encode_sample(self, x):
        """Кодирование одного отсчета"""
        # Предсказание
        y_pred = np.sum(self.a * self.buffer)
        z = x - y_pred
        
        # Обновление коэффициентов по правилам (8)
        abs_buffer = np.abs(self.buffer)
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            # Обновляем только один коэффициент
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        
        return z
    
    def decode_sample(self, z):
        """Декодирование одного отсчета"""
        # Восстановление
        y_pred = np.sum(self.a * self.buffer)
        x_rec = z + y_pred
        
        # ТОЧНО ТАКОЕ ЖЕ обновление коэффициентов!
        abs_buffer = np.abs(self.buffer)
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x_rec
        
        return x_rec

class ImprovedL1Codec(L1Codec):
    """Улучшенный L₁-алгоритм с попарными полусумами"""
    def __init__(self, n=4, L=16, method='median'):
        super().__init__(n, L)
        self.method = method  # 'median' или 'all_pairs'
    
    def process_buffer(self, buffer):
        """Обработка буфера методом попарных полусумм"""
        n = len(buffer)
        
        if self.method == 'median':
            # МЕТОД 1: Медиана попарных полусумм (симметричные пары)
            if n >= 2:
                # Формируем пары от краев к центру
                pairs = []
                for j in range(n // 2):
                    left = buffer[j]
                    right = buffer[n - 1 - j]
                    pairs.append((left + right) / 2.0)
                
                if pairs:
                    median_val = np.median(pairs)
                    # Создаем новый буфер с медианным значением
                    processed = np.full_like(buffer, median_val)
                    return processed
            
            # Если не можем обработать, возвращаем исходный буфер
            return np.copy(buffer)
            
        elif self.method == 'all_pairs':
            # МЕТОД 2: Все возможные пары (ваш метод)
            if n <= 1:
                return np.copy(buffer)
            
            # Создаем список всех значений
            all_values = list(buffer)
            
            # Добавляем все попарные полусуммы
            for i in range(n):
                for j in range(i + 1, n):
                    all_values.append((buffer[i] + buffer[j]) / 2.0)
            
            # Вычисляем медиану всех этих значений
            median_val = np.median(all_values)
            
            # Создаем новый буфер с медианным значением
            processed = np.full_like(buffer, median_val)
            
            return processed
            
        else:
            # Без обработки (как базовый алгоритм)
            return np.copy(buffer)
    
    def encode_sample(self, x):
        """Кодирование с предобработкой буфера"""
        # Предварительная обработка буфера
        processed_buffer = self.process_buffer(self.buffer)
        
        # Предсказание на обработанном буфере
        y_pred = np.sum(self.a * processed_buffer)
        z = x - y_pred
        
        # ОБРАТИТЕ ВНИМАНИЕ: обновление коэффициентов использует ИСХОДНЫЙ буфер
        # Это важно для сохранения логики L₁-алгоритма
        abs_buffer = np.abs(self.buffer)  # Исходный буфер!
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера (записываем исходный x)
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        
        return z
    
    def decode_sample(self, z):
        """Декодирование с предобработкой буфера"""
        # Предварительная обработка буфера
        processed_buffer = self.process_buffer(self.buffer)
        
        # Восстановление на обработанном буфере
        y_pred = np.sum(self.a * processed_buffer)
        x_rec = z + y_pred
        
        # ТОЧНО ТАКОЕ ЖЕ обновление коэффициентов
        abs_buffer = np.abs(self.buffer)
        max_idx = np.argmax(abs_buffer)
        max_val = abs_buffer[max_idx]
        
        if max_val > self.L and max_val != 0:
            self.a[max_idx] += z / max_val
        
        # Сдвиг буфера
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x_rec
        
        return x_rec

# ТЕСТ: Сравнение алгоритмов
def compare_algorithms():
    print("СРАВНЕНИЕ АЛГОРИТМОВ L₁")
    print("="*60)
    
    # Создаем тестовый сигнал с помехами
    np.random.seed(42)
    n_samples = 1000
    t = np.linspace(0, 1, n_samples)
    
    # Чистый сигнал
    clean_signal = 50 * np.sin(2 * np.pi * 3 * t) + 30 * np.sin(2 * np.pi * 10 * t)
    
    # Добавляем импульсные помехи
    signal = clean_signal.copy()
    noise_positions = [150, 300, 450, 600, 750]
    for pos in noise_positions:
        signal[pos] += np.random.uniform(100, 200)
    
    # Добавляем небольшую случайную помеху
    signal += np.random.normal(0, 5, n_samples)
    
    # Инициализируем кодексы
    codecs = {
        'Базовый L₁': L1Codec(n=4, L=16),
        'Медиана полусумм': ImprovedL1Codec(n=4, L=16, method='median'),
        'Все пары': ImprovedL1Codec(n=4, L=16, method='all_pairs')
    }
    
    results = {}
    
    for name, codec in codecs.items():
        print(f"\nТестируем: {name}")
        
        # Кодирование
        encoded = []
        for i, x in enumerate(signal):
            z = codec.encode_sample(x)
            encoded.append(z)
        
        # Сброс кодера для декодирования
        if name == 'Базовый L₁':
            codec = L1Codec(n=4, L=16)
        elif 'Медиана' in name:
            codec = ImprovedL1Codec(n=4, L=16, method='median')
        else:
            codec = ImprovedL1Codec(n=4, L=16, method='all_pairs')
        
        # Декодирование
        decoded = []
        for z in encoded:
            x_rec = codec.decode_sample(z)
            decoded.append(x_rec)
        
        decoded = np.array(decoded)
        
        # Вычисляем метрики (пропускаем первые n отсчетов)
        start_idx = 4
        mse = np.mean((signal[start_idx:] - decoded[start_idx:])**2)
        max_error = np.max(np.abs(signal[start_idx:] - decoded[start_idx:]))
        std_z = np.std(encoded[start_idx:])
        var_z = np.var(encoded[start_idx:])
        
        # Энтропия (приближенно)
        hist, bins = np.histogram(encoded[start_idx:], bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist)) * (bins[1] - bins[0])
        
        results[name] = {
            'mse': mse,
            'max_error': max_error,
            'std_z': std_z,
            'var_z': var_z,
            'entropy': entropy,
            'encoded': np.array(encoded),
            'decoded': decoded
        }
        
        print(f"  Дисперсия z: {var_z:.2f}")
        print(f"  Энтропия z: {entropy:.3f} бит")
        print(f"  Макс. ошибка: {max_error:.2e}")
        print(f"  MSE: {mse:.2e}")
        
        # Проверка lossless
        is_lossless = np.allclose(signal[start_idx:], decoded[start_idx:], atol=1e-12)
        print(f"  Lossless: {'ДА' if is_lossless else 'НЕТ'}")
    
    return signal, results

# Визуализация результатов
def visualize_comparison(signal, results):
    """Визуализация сравнения алгоритмов"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    colors = {
        'Базовый L₁': 'red',
        'Медиана полусумм': 'blue',
        'Все пары': 'green'
    }
    
    # 1. Исходный сигнал с помехами
    ax = axes[0, 0]
    ax.plot(signal, 'k-', linewidth=1, alpha=0.7, label='Сигнал')
    ax.set_title('Исходный сигнал с импульсными помехами')
    ax.set_xlabel('Отсчет')
    ax.set_ylabel('Амплитуда')
    ax.grid(True, alpha=0.3)
    
    # Отметим помехи
    noise_positions = [150, 300, 450, 600, 750]
    for pos in noise_positions:
        ax.axvline(x=pos, color='r', linestyle='--', alpha=0.3)
    
    # 2. Сжатые сигналы (z)
    ax = axes[0, 1]
    for name, res in results.items():
        ax.plot(np.abs(res['encoded'][4:]), 
                color=colors[name], 
                alpha=0.6, 
                linewidth=0.8,
                label=f"{name}")
    ax.set_title('Абсолютные значения сжатого сигнала |z|')
    ax.set_xlabel('Отсчет')
    ax.set_ylabel('|z|')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    
    # 3. Гистограмма распределения z
    ax = axes[1, 0]
    for name, res in results.items():
        ax.hist(res['encoded'][4:], 
                bins=50, 
                alpha=0.5, 
                color=colors[name],
                label=name,
                density=True,
                histtype='stepfilled')
    ax.set_title('Распределение значений z')
    ax.set_xlabel('z')
    ax.set_ylabel('Плотность вероятности')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # 4. Сравнение дисперсии
    ax = axes[1, 1]
    names = list(results.keys())
    var_values = [results[name]['var_z'] for name in names]
    
    bars = ax.bar(names, var_values, color=[colors[n] for n in names])
    ax.set_title('Дисперсия сжатого сигнала Var[z]\n(меньше = лучше сжатие)')
    ax.set_ylabel('Дисперсия')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, var_val in zip(bars, var_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{var_val:.1f}',
                ha='center', va='bottom')
    
    # 5. Сравнение энтропии
    ax = axes[2, 0]
    entropy_values = [results[name]['entropy'] for name in names]
    
    bars = ax.bar(names, entropy_values, color=[colors[n] for n in names])
    ax.set_title('Энтропия сжатого сигнала\n(меньше = лучше сжатие)')
    ax.set_ylabel('Энтропия, бит')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, entropy_val in zip(bars, entropy_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{entropy_val:.3f}',
                ha='center', va='bottom')
    
    # 6. Улучшение относительно базового
    ax = axes[2, 1]
    base_var = results['Базовый L₁']['var_z']
    
    improvements = []
    for name in names:
        if name != 'Базовый L₁':
            improv = (base_var - results[name]['var_z']) / base_var * 100
            improvements.append(improv)
    
    if improvements:
        ax_names = [n for n in names if n != 'Базовый L₁']
        bars = ax.bar(ax_names, improvements, 
                     color=[colors[n] for n in ax_names])
        ax.set_title('Уменьшение дисперсии относительно базового')
        ax.set_ylabel('Улучшение, %')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, improv in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{improv:.1f}%',
                   ha='center', va='bottom')
        
        # Среднее улучшение
        avg_improvement = np.mean(improvements)
        ax.axhline(y=avg_improvement, color='r', linestyle='--', alpha=0.5)
        ax.text(0.5, avg_improvement + 1, f'Среднее: {avg_improvement:.1f}%',
                ha='center', va='bottom', color='r')
    
    plt.tight_layout()
    plt.savefig('l1_comparison_corrected.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Вывод итоговых результатов
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("="*60)
    
    base_var = results['Базовый L₁']['var_z']
    base_entropy = results['Базовый L₁']['entropy']
    
    for name in names:
        if name != 'Базовый L₁':
            var_improv = (base_var - results[name]['var_z']) / base_var * 100
            entropy_improv = (base_entropy - results[name]['entropy']) / base_entropy * 100
            
            print(f"\n{name}:")
            print(f"  Уменьшение дисперсии: {var_improv:.1f}%")
            print(f"  Уменьшение энтропии: {entropy_improv:.1f}%")
            print(f"  Эффективность сжатия: {(var_improv + entropy_improv)/2:.1f}%")

# Тест производительности
def test_performance_corrected():
    """Тест производительности исправленных алгоритмов"""
    print("\n" + "="*60)
    print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    
    # Большой сигнал для теста
    n_samples = 50000  # Уменьшим для скорости
    signal = np.random.normal(0, 100, n_samples)
    
    algorithms = [
        ('Базовый L₁', lambda: L1Codec(n=4, L=16)),
        ('Медиана полусумм', lambda: ImprovedL1Codec(n=4, L=16, method='median')),
        ('Все пары', lambda: ImprovedL1Codec(n=4, L=16, method='all_pairs'))
    ]
    
    for name, init_func in algorithms:
        # Тест кодирования
        codec = init_func()
        start_time = time.time()
        
        encoded = []
        for x in signal:
            encoded.append(codec.encode_sample(x))
        
        encode_time = time.time() - start_time
        
        # Тест декодирования
        codec = init_func()
        start_time = time.time()
        
        decoded = []
        for z in encoded:
            decoded.append(codec.decode_sample(z))
        
        decode_time = time.time() - start_time
        
        # Проверка lossless
        decoded = np.array(decoded)
        is_lossless = np.allclose(signal[4:], decoded[4:], atol=1e-10)
        
        print(f"\n{name}:")
        print(f"  Кодирование: {encode_time:.3f} сек ({n_samples/encode_time:.0f} отсч/сек)")
        print(f"  Декодирование: {decode_time:.3f} сек")
        print(f"  Lossless: {'ДА' if is_lossless else 'НЕТ'}")

# Упрощенный тест для быстрой проверки
def quick_test():
    """Быстрый тест без визуализации"""
    print("БЫСТРЫЙ ТЕСТ L₁-АЛГОРИТМОВ")
    print("="*60)
    
    # Простой тестовый сигнал
    t = np.linspace(0, 0.1, 500)
    signal = 100 * np.sin(2 * np.pi * 5 * t)
    
    # Тестируем базовый алгоритм
    base_codec = L1Codec(n=4, L=16)
    encoded_base = [base_codec.encode_sample(x) for x in signal]
    
    # Сбрасываем
    base_codec = L1Codec(n=4, L=16)
    decoded_base = [base_codec.decode_sample(z) for z in encoded_base]
    
    # Тестируем улучшенный алгоритм (все пары)
    improved_codec = ImprovedL1Codec(n=4, L=16, method='all_pairs')
    encoded_improved = [improved_codec.encode_sample(x) for x in signal]
    
    # Сбрасываем
    improved_codec = ImprovedL1Codec(n=4, L=16, method='all_pairs')
    decoded_improved = [improved_codec.decode_sample(z) for z in encoded_improved]
    
    # Сравниваем
    decoded_base = np.array(decoded_base)
    decoded_improved = np.array(decoded_improved)
    
    print("Проверка lossless-свойства:")
    is_base_lossless = np.allclose(signal[4:], decoded_base[4:], atol=1e-10)
    is_improved_lossless = np.allclose(signal[4:], decoded_improved[4:], atol=1e-10)
    print(f"  Базовый алгоритм: {'ДА' if is_base_lossless else 'НЕТ'}")
    print(f"  Улучшенный алгоритм: {'ДА' if is_improved_lossless else 'НЕТ'}")
    
    print("\nСравнение дисперсии сжатого сигнала:")
    var_base = np.var(encoded_base[4:])
    var_improved = np.var(encoded_improved[4:])
    improvement = (var_base - var_improved) / var_base * 100
    
    print(f"  Базовый: Var[z] = {var_base:.2f}")
    print(f"  Улучшенный: Var[z] = {var_improved:.2f}")
    print(f"  Улучшение: {improvement:.1f}%")
    
    return improvement > 0

# Основная функция
if __name__ == "__main__":
    print("ИСПРАВЛЕННАЯ РЕАЛИЗАЦИЯ L₁-АЛГОРИТМОВ С ПОПАРНЫМИ ПОЛУСУММАМИ")
    print("="*60)
    
    # Быстрый тест для проверки
    if quick_test():
        print("\nБыстрый тест пройден успешно! Запускаем полное сравнение...")
        
        # Запускаем полное сравнение
        signal, results = compare_algorithms()
        
        # Визуализация
        visualize_comparison(signal, results)
        
        # Тест производительности
        test_performance_corrected()
    else:
        print("\nОшибка в быстром тесте! Проверьте реализацию алгоритма.")
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)
