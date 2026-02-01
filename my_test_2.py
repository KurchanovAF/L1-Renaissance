# test_speech.py
import numpy as np
import sys
sys.path.insert(0, 'src')

from core_2006 import L1AdaptiveFilter2006

# Создаем структурированный сигнал (имитация речи)
np.random.seed(42)
n_samples = 8000  # 1 секунда при 8 кГц

# AR модель как в вашей статье
signal = np.random.randn(n_samples)
for i in range(4, n_samples):
    # AR(4) процесс с коэффициентами близкими к реальной речи
    signal[i] += 0.8*signal[i-1] - 0.5*signal[i-2] + 0.3*signal[i-3] - 0.2*signal[i-4]

filter = L1AdaptiveFilter2006(order=4, mu=0.03)
compressed, coeffs = filter.process(signal)

original_variance = np.var(signal[4:])
compressed_variance = np.var(compressed)
ratio = compressed_variance / original_variance

print(f"Тест на структурированном сигнале (имитация речи):")
print(f"  Дисперсия исходного: {original_variance:.4f}")
print(f"  Дисперсия сжатого:   {compressed_variance:.4f}")
print(f"  Коэффициент сжатия:  {ratio:.4f}")
print(f"  Уменьшение на:       {(1-ratio)*100:.1f}%")
