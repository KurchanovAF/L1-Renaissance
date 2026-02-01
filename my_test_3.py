# compare_with_lms.py
import numpy as np
import sys
sys.path.insert(0, 'src')

from core_2006 import L1AdaptiveFilter2006

class LMSFilter:
    """Классический LMS фильтр для сравнения"""
    def __init__(self, order=4, mu=0.01):
        self.order = order
        self.mu = mu
        self.coeffs = np.zeros(order)
    
    def process(self, signal):
        n = len(signal)
        compressed = np.zeros(n - self.order)
        
        for i in range(self.order, n):
            x_past = signal[i-self.order:i][::-1]
            prediction = np.dot(self.coeffs, x_past)
            error = signal[i] - prediction
            
            # LMS update (L2 metric)
            self.coeffs += self.mu * error * x_past
            compressed[i-self.order] = error
            
        return compressed

# Тест сравнения
np.random.seed(42)
signal = np.random.randn(2000)

# Ваш L1 алгоритм
l1_filter = L1AdaptiveFilter2006(order=4, mu=0.03)
l1_compressed = l1_filter.process(signal)[0]
l1_ratio = np.var(l1_compressed) / np.var(signal[4:])

# Классический LMS
lms_filter = LMSFilter(order=4, mu=0.01)
lms_compressed = lms_filter.process(signal)
lms_ratio = np.var(lms_compressed) / np.var(signal[4:])

print("СРАВНЕНИЕ L1 vs LMS:")
print(f"  L1 алгоритм (2006):  {l1_ratio:.4f}")
print(f"  Классический LMS:    {lms_ratio:.4f}")
print(f"  L1 лучше на:         {(lms_ratio - l1_ratio)*100:.1f}%")
