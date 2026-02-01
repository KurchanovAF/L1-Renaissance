# test_direct.py - положить в корень проекта
import sys
sys.path.insert(0, 'src')  # добавляем src в путь

from core_2006 import L1AdaptiveFilter2006
import numpy as np

print("Тестирую алгоритм напрямую...")
filter = L1AdaptiveFilter2006()
signal = np.random.randn(1000)
compressed, _ = filter.process(signal)
print(f"✅ Алгоритм работает! Коэффициент: {np.var(compressed)/np.var(signal[4:]):.4f}")
