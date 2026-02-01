# test_exact_reconstruction.py
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("ТЕСТ ТОЧНОГО ВОССТАНОВЛЕНИЯ (без scipy)")
print("="*70)

# Простая реализация без scipy
class L1Codec2006Simple:
    """
    Упрощенная, но правильная реализация по смыслу статьи.
    Все вычисления с учетом целочисленности.
    """
    
    def __init__(self, order=4, bt=16.0):
        self.order = order
        self.bt = bt
        self.coeffs = np.zeros(order, dtype=np.float64)
        
    def encode_sample(self, x_int, past_int):
        """
        Кодирование одного отсчета.
        
        Args:
            x_int: текущий отсчет (целое -128..127)
            past_int: past samples [x[n-1], ..., x[n-order]] (целые)
            
        Returns:
            z_int: сжатый отсчет (целое)
        """
        # 1. Предсказание
        prediction = np.dot(self.coeffs, past_int)  # float
        
        # 2. y = 256*x - prediction (<<8 = *256)
        y_float = 256.0 * x_int - prediction
        
        # 3. Округление до целого (КАК В СТАТЬЕ!)
        if y_float >= 0:
            z_int = int((y_float + 127) // 256)
        else:
            z_int = -int(((-y_float + 128) // 256))
        
        # 4. Обновление коэффициентов (упрощенно, но по смыслу)
        h1 = y_float  # то же что y_float
        
        # Находим максимальный |past|
        abs_past = np.abs(past_int)
        max_idx = np.argmax(abs_past)
        xmax = abs_past[max_idx]
        
        if xmax > 0:
            # Упрощенное обновление (в духе статьи)
            self.coeffs[max_idx] += np.sign(h1) * np.sign(past_int[max_idx]) * 0.03
        
        return z_int
    
    def decode_sample(self, z_int, past_int):
        """
        Декодирование одного отсчета.
        
        Args:
            z_int: сжатый отсчет
            past_int: уже восстановленные past samples
            
        Returns:
            x_rec_int: восстановленный отсчет
        """
        # 1. Предсказание (те же коэффициенты!)
        prediction = np.dot(self.coeffs, past_int)
        
        # 2. Восстановление x из z
        # z = round((256*x - prediction)/256)
        # => 256*x ≈ 256*z + prediction
        # => x ≈ z + prediction/256
        
        x_rec_float = z_int + prediction / 256.0
        
        # 3. Округляем до целого (как исходный АЦП)
        x_rec_int = int(np.round(x_rec_float))
        
        # 4. ОБНОВЛЯЕМ коэффициенты ТАК ЖЕ как при кодировании
        #    используя восстановленный x
        y_float_rec = 256.0 * x_rec_int - prediction
        h1_rec = y_float_rec
        
        abs_past = np.abs(past_int)
        max_idx = np.argmax(abs_past)
        xmax = abs_past[max_idx]
        
        if xmax > 0:
            self.coeffs[max_idx] += np.sign(h1_rec) * np.sign(past_int[max_idx]) * 0.03
        
        return x_rec_int

# ТЕСТ 1: Проверка округления
print("\n1. ТЕСТ ОКРУГЛЕНИЯ (как в статье):")
print("-"*40)

test_values = [-300, -255, -128, -1, 0, 1, 127, 255, 300]
for y in test_values:
    if y >= 0:
        z = (y + 127) // 256
    else:
        z = -((-y + 128) // 256)
    
    # Обратно (приблизительно)
    y_rec = 256 * z
    
    print(f"  y={y:4d} -> z={z:3d} -> y'={y_rec:4d}, ошибка={y_rec-y:4d}")

# ТЕСТ 2: Полный цикл кодирование-декодирование
print("\n\n2. ПОЛНЫЙ ЦИКЛ КОДИРОВАНИЯ-ДЕКОДИРОВАНИЯ:")
print("-"*40)

# Создаем тестовый сигнал (целые числа как из 8-бит АЦП)
np.random.seed(42)
n_samples = 100
original = np.random.randint(-128, 128, n_samples, dtype=np.int32)  # -128..127

print(f"Исходный сигнал: {len(original)} samples")
print(f"  Диапазон: [{original.min()}, {original.max()}]")
print(f"  Пример: {original[:10]}")

# КОДЕР
encoder = L1Codec2006Simple(order=4, bt=16.0)
encoded = []

for i in range(4, n_samples):
    x = original[i]
    past = original[i-4:i][::-1]  # [x[n-1], x[n-2], x[n-3], x[n-4]]
    z = encoder.encode_sample(x, past)
    encoded.append(z)

encoded = np.array(encoded, dtype=np.int32)
print(f"\nЗакодированный: {len(encoded)} values")
print(f"  Диапазон: [{encoded.min()}, {encoded.max()}]")
print(f"  Пример: {encoded[:10]}")

# ДЕКОДЕР (с теми же начальными коэффициентами!)
decoder = L1Codec2006Simple(order=4, bt=16.0)
decoder.coeffs = encoder.coeffs.copy()  # важно!

# Начинаем с первых 4 samples (как в статье)
decoded = original[:4].copy().tolist()

for i in range(len(encoded)):
    z = encoded[i]
    past = decoded[-4:]  # последние 4 восстановленных
    x_rec = decoder.decode_sample(z, past)
    decoded.append(x_rec)

decoded = np.array(decoded, dtype=np.int32)

print(f"\nДекодированный: {len(decoded)} samples")
print(f"  Пример: {decoded[:10]}")

# ПРОВЕРКА
print("\n3. ПРОВЕРКА ТОЧНОСТИ:")
print("-"*40)

if len(original) == len(decoded):
    exact_match = np.array_equal(original, decoded)
    
    print(f"Длины: оригинал={len(original)}, декодирован={len(decoded)}")
    print(f"Побитовое совпадение: {'✅ ДА' if exact_match else '❌ НЕТ'}")
    
    if not exact_match:
        diff = original - decoded
        diff_indices = np.where(diff != 0)[0]
        
        print(f"Не совпадают в {len(diff_indices)} позициях из {len(original)}")
        print(f"Максимальная разница: {np.max(np.abs(diff))}")
        
        # Покажем первые различия
        print("\nПервые 5 различий:")
        for idx in diff_indices[:5]:
            print(f"  x[{idx}] = {original[idx]}, x_rec[{idx}] = {decoded[idx]}, diff = {diff[idx]}")
else:
    print(f"❌ Разная длина! оригинал={len(original)}, декодирован={len(decoded)}")

# ТЕСТ 3: Почему может не совпадать?
print("\n\n4. АНАЛИЗ ПРИЧИН НЕТОЧНОСТИ:")
print("-"*40)

# Главная проблема: при декодировании мы используем УЖЕ ДЕКОДИРОВАННЫЕ
# past samples, которые могут немного отличаться от оригинальных!
# Это накапливает ошибку.

print("Проблема: при декодировании используются:")
print("  - Не оригинальные past samples")
print("  - А уже декодированные (возможно с ошибкой)")
print("  - Ошибка накапливается!")

# ТЕСТ 4: Идеальный случай (используем оригинальные past при декодировании)
print("\n\n5. ТЕСТ 'ИДЕАЛЬНОГО' ДЕКОДЕРА (использует оригинальные past):")
print("-"*40)

decoder_ideal = L1Codec2006Simple(order=4, bt=16.0)
decoder_ideal.coeffs = encoder.coeffs.copy()

decoded_ideal = original[:4].copy().tolist()

for i in range(len(encoded)):
    z = encoded[i]
    # ИДЕАЛЬНЫЙ СЛУЧАЙ: используем ОРИГИНАЛЬНЫЕ past!
    past_ideal = original[i:i+4][::-1] if i+4 <= len(original) else decoded_ideal[-4:]
    x_rec = decoder_ideal.decode_sample(z, past_ideal)
    decoded_ideal.append(x_rec)

decoded_ideal = np.array(decoded_ideal, dtype=np.int32)

exact_match_ideal = np.array_equal(original, decoded_ideal)
print(f"С оригинальными past: {'✅ ТОЧНОЕ совпадение!' if exact_match_ideal else '❌ Есть ошибки'}")

if not exact_match_ideal:
    diff_ideal = original - decoded_ideal
    print(f"  Ошибок: {np.sum(diff_ideal != 0)}")
    print(f"  Макс ошибка: {np.max(np.abs(diff_ideal))}")

# ВИЗУАЛИЗАЦИЯ
print("\n\n6. ВИЗУАЛИЗАЦИЯ:")
print("-"*40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Оригинальный сигнал
axes[0, 0].plot(original, 'b-', linewidth=2, alpha=0.7, label='Оригинал')
axes[0, 0].plot(decoded, 'r--', linewidth=1, alpha=0.7, label='Декодированный')
axes[0, 0].set_title('Оригинальный vs Декодированный сигнал')
axes[0, 0].set_xlabel('Отсчет')
axes[0, 0].set_ylabel('Амплитуда')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Ошибка декодирования
error = original - decoded
axes[0, 1].plot(error, 'g-', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[0, 1].set_title('Ошибка декодирования')
axes[0, 1].set_xlabel('Отсчет')
axes[0, 1].set_ylabel('Ошибка')
axes[0, 1].grid(True, alpha=0.3)

# 3. Закодированный сигнал (ошибки предсказания)
axes[1, 0].plot(encoded, 'purple', alpha=0.7)
axes[1, 0].set_title('Закодированный сигнал (ошибки предсказания)')
axes[1, 0].set_xlabel('Отсчет')
axes[1, 0].set_ylabel('z[n]')
axes[1, 0].grid(True, alpha=0.3)

# 4. Распределение ошибок
axes[1, 1].hist(error, bins=20, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title('Распределение ошибок декодирования')
axes[1, 1].set_xlabel('Ошибка')
axes[1, 1].set_ylabel('Частота')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('ТЕСТ ПОЛНОГО ЦИКЛА: Кодирование → Декодирование', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Сохраняем и показываем
plt.savefig('full_cycle_test_simple.png', dpi=150, bbox_inches='tight')
print("График сохранен как 'full_cycle_test_simple.png'")
plt.show()

# ВЫВОДЫ
print("\n" + "="*70)
print("ВАЖНЫЕ ВЫВОДЫ:")
print("="*70)
print("1. ✅ Алгоритм из статьи 2006 ДОЛЖЕН быть lossless")
print("2. ⚠️  Но для этого нужны:")
print("   - Точная реализация всех формул")
print("   - Одинаковое обновление коэффициентов на кодере и декодере")
print("   - Использование ЦЕЛЫХ чисел и правильного округления")
print("3. 🔧 Наша упрощенная реализация показывает:")
print("   - Принцип работы понятен")
print("   - Но могут быть накапливающиеся ошибки")
print("   - Нужна ТОЧНАЯ реализация по статьям")
print("\nСледующий шаг: реализовать ТОЧНО по формулам из статьи!")
print("="*70)
