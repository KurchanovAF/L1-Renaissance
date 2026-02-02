import numpy as np

# САМАЯ ПРОСТАЯ, НО ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ
class SimpleCorrectL1:
    """Минимальная, но правильная реализация L₁-алгоритма"""
    def __init__(self, n=4, L=30):
        self.n = n
        self.L = L
        self.a = np.zeros(n, dtype=np.float64)  # коэффициенты
        self.x_buffer = np.zeros(n, dtype=np.float64)  # история x
    
    def process(self, x, is_encode=True, z_input=None):
        """
        Универсальная обработка одного отсчета
        Возвращает z при кодировании, x при декодировании
        """
        # 1. Предсказание: x̂ = Σ a_k * x_{i-k}
        pred = np.sum(self.a * self.x_buffer)
        
        if is_encode:
            # КОДИРОВАНИЕ: z = x - x̂
            z = x - pred
            z_int = int(round(z))
            current_x = x
        else:
            # ДЕКОДИРОВАНИЕ: x = z + x̂
            z_int = z_input
            x_rec = z_int + pred
            current_x = x_rec
        
        # 2. Находим X = max(|x_{i-k}|)
        max_val = 0
        max_idx = 0
        
        for k in range(self.n):
            abs_val = abs(self.x_buffer[k])
            if abs_val > max_val:
                max_val = abs_val
                max_idx = k
        
        # 3. Обновление коэффициентов (ТОЛЬКО если max_val > L)
        if max_val > self.L and max_val != 0:
            # Обновляем ОДИН коэффициент
            delta = z_int / max_val
            
            # СИЛЬНОЕ ограничение для стабильности
            if abs(delta) > 0.1:  # ОЧЕНЬ МАЛОЕ ОБНОВЛЕНИЕ!
                delta = np.sign(delta) * 0.1
            
            self.a[max_idx] += delta
            
            # Ограничиваем коэффициенты
            if abs(self.a[max_idx]) > 2.0:
                self.a[max_idx] = np.sign(self.a[max_idx]) * 2.0
        
        # 4. Сдвиг буфера
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = current_x
        
        if is_encode:
            return z_int
        else:
            return current_x

# ПРОСТЕЙШИЙ ТЕСТ
def simplest_test():
    print("="*60)
    print("САМЫЙ ПРОСТОЙ ТЕСТ")
    print("="*60)
    
    # Простейший сигнал
    signal = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
    print(f"Сигнал: {signal}")
    print(f"Дисперсия сигнала: {np.var(signal):.2f}")
    
    # Кодирование
    encoder = SimpleCorrectL1(n=4, L=30)
    encoded = []
    
    for x in signal:
        z = encoder.process(x, is_encode=True)
        encoded.append(z)
    
    encoded = np.array(encoded)
    print(f"\nЗакодировано z: {encoded}")
    print(f"Дисперсия z: {np.var(encoded):.2f}")
    
    # Декодирование
    decoder = SimpleCorrectL1(n=4, L=30)
    decoded = []
    
    for z in encoded:
        x_rec = decoder.process(None, is_encode=False, z_input=z)
        decoded.append(x_rec)
    
    decoded = np.array(decoded)
    print(f"\nВосстановлено: {decoded}")
    
    # Проверка
    errors = np.abs(signal - decoded)
    print(f"Ошибки: {errors}")
    print(f"Макс ошибка: {np.max(errors):.2e}")
    
    if np.all(errors == 0):
        print("✓ ИДЕАЛЬНОЕ ВОССТАНОВЛЕНИЕ!")
    else:
        print("✗ Есть ошибки")
    
    return signal, encoded, decoded, errors

# Тест с МАЛЕНЬКИМИ значениями
def small_values_test():
    print("\n" + "="*60)
    print("ТЕСТ С МАЛЕНЬКИМИ ЗНАЧЕНИЯМИ (как 8-bit речь)")
    print("="*60)
    
    # Сигнал в диапазоне ±20 (очень маленький)
    signal = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], dtype=np.int32)
    print(f"Сигнал: {signal}")
    print(f"Дисперсия: {np.var(signal):.2f}")
    
    encoder = SimpleCorrectL1(n=4, L=30)
    encoded = []
    
    for x in signal:
        z = encoder.process(x, is_encode=True)
        encoded.append(z)
    
    encoded = np.array(encoded)
    print(f"\nz: {encoded}")
    print(f"Дисперсия z: {np.var(encoded):.2f}")
    
    # Проверяем уменьшение дисперсии
    if np.var(encoded) < np.var(signal):
        reduction = (1 - np.var(encoded)/np.var(signal)) * 100
        print(f"✓ Дисперсия уменьшилась на {reduction:.1f}%")
    else:
        print(f"✗ Дисперсия увеличилась!")
    
    return signal, encoded

# Тест с нулевым сигналом
def zero_signal_test():
    print("\n" + "="*60)
    print("ТЕСТ С НУЛЕВЫМ СИГНАЛОМ (самый простой)")
    print("="*60)
    
    signal = np.zeros(10, dtype=np.int32)
    print(f"Сигнал: {signal}")
    
    encoder = SimpleCorrectL1(n=4, L=30)
    encoded = []
    
    for x in signal:
        z = encoder.process(x, is_encode=True)
        encoded.append(z)
    
    encoded = np.array(encoded)
    print(f"z: {encoded}")
    print(f"Все z должны быть 0: {np.all(encoded == 0)}")
    
    if np.all(encoded == 0):
        print("✓ Алгоритм работает правильно для нулевого сигнала")
    else:
        print("✗ Проблема с нулевым сигналом")

# Покажем, ЧТО именно происходит в алгоритме
def debug_step_by_step():
    print("\n" + "="*60)
    print("ПОШАГОВАЯ ОТЛАДКА")
    print("="*60)
    
    # Всего 5 отсчетов
    signal = [10, 20, 30, 40, 50]
    
    encoder = SimpleCorrectL1(n=4, L=30)
    
    print("Шаг |  x | Буфер x | Коэфф a | Предск. |  z | Обновление")
    print("-" * 70)
    
    for i, x in enumerate(signal):
        # Вручную вычисляем предсказание
        pred = np.sum(encoder.a * encoder.x_buffer)
        z = x - pred
        z_int = int(round(z))
        
        # Находим максимальный в буфере
        max_val = 0
        max_idx = 0
        for k in range(encoder.n):
            abs_val = abs(encoder.x_buffer[k])
            if abs_val > max_val:
                max_val = abs_val
                max_idx = k
        
        # Показываем состояние ДО обновления
        print(f"{i:3d} | {x:3d} | {encoder.x_buffer} | {encoder.a} | {pred:7.1f} | {z_int:3d} | ", end="")
        
        # Обновляем
        if max_val > encoder.L and max_val != 0:
            delta = z_int / max_val
            if abs(delta) > 0.1:
                delta = np.sign(delta) * 0.1
            encoder.a[max_idx] += delta
            print(f"a[{max_idx}]+={delta:.3f}")
        else:
            print("нет обновления")
        
        # Сдвигаем буфер
        encoder.x_buffer = np.roll(encoder.x_buffer, 1)
        encoder.x_buffer[0] = x

# Тест с реальными значениями из статьи
def article_parameters_test():
    print("\n" + "="*60)
    print("ТЕСТ С ПАРАМЕТРАМИ ИЗ СТАТЬИ")
    print("="*60)
    
    # Параметры точно как в статье
    FS = 8000  # 8 кГц
    BIT_DEPTH = 8  # 8-bit
    
    # Сигнал в диапазоне 8-bit
    n_samples = 50
    signal = np.random.randint(-127, 128, n_samples)
    
    print(f"Сигнал: {n_samples} отсчетов 8-bit")
    print(f"Диапазон: [{signal.min()}, {signal.max()}]")
    print(f"Дисперсия исходного: {np.var(signal):.2f}")
    
    # Пробуем разные L
    print("\nТестируем разные L:")
    print(" L | var(z) | var(z)/var(x) | Коэффициенты (после обработки)")
    print("-" * 70)
    
    for L in [10, 20, 30, 40]:
        encoder = SimpleCorrectL1(n=4, L=L)
        encoded = []
        
        for x in signal:
            encoded.append(encoder.process(x, is_encode=True))
        
        encoded = np.array(encoded)
        var_ratio = np.var(encoded) / np.var(signal)
        
        print(f"{L:2d} | {np.var(encoded):7.2f} | {var_ratio:13.3f} | {encoder.a}")

if __name__ == "__main__":
    print("МИНИМАЛЬНАЯ, НО ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ L₁")
    print("Цель: показать, что дисперсия z МЕНЬШЕ дисперсии x")
    print("="*60)
    
    # 1. Самый простой тест
    signal1, encoded1, decoded1, errors1 = simplest_test()
    
    # 2. Тест с маленькими значениями
    signal2, encoded2 = small_values_test()
    
    # 3. Тест с нулевым сигналом
    zero_signal_test()
    
    # 4. Пошаговая отладка
    debug_step_by_step()
    
    # 5. Тест с параметрами из статьи
    article_parameters_test()
    
    print("\n" + "="*60)
    print("АНАЛИЗ ПРОБЛЕМ:")
    print("="*60)
    print("Если дисперсия z БОЛЬШЕ дисперсии x, то:")
    print("1. Возможно, L слишком маленькое (коэффициенты меняются слишком часто)")
    print("2. Возможно, обновление коэффициентов слишком большое")
    print("3. Возможно, нужно начинать с ненулевых коэффициентов")
    
    print("\nРЕКОМЕНДАЦИИ:")
    print("1. Уменьшить максимальное обновление (delta)")
    print("2. Увеличить L")
    print("3. Использовать предобученные начальные коэффициенты")
