"""
Improved L1 adaptive filter using pairwise medians.
Based on the idea from 2024 discussions.
"""

import numpy as np
from typing import Tuple, Optional


class PairwiseMedianL1Filter:
    """
    Enhanced L1 adaptive filter using median of pairwise sums.
    
    Improvement over original 2006 algorithm:
    - Uses information from multiple samples via pairwise combinations
    - More stable convergence
    - Maintains robustness to outliers
    
    Parameters:
        order: Filter order (default 4)
        mu: Learning rate (default 0.03)
        use_median: Whether to use median or mean of pairwise sums
    """
    
    def __init__(self, order: int = 4, mu: float = 0.03, use_median: bool = True):
        self.order = order
        self.mu = mu
        self.use_median = use_median
        self.coeffs = np.zeros(order, dtype=np.float64)
        
    def reset(self) -> None:
        """Reset filter coefficients."""
        self.coeffs = np.zeros(self.order, dtype=np.float64)
        
    def _select_pair(self, x_past: np.ndarray) -> Tuple[int, int, float]:
        """
        Select the best pair of indices using pairwise combinations.
        
        Args:
            x_past: Past samples [x[n-1], x[n-2], ..., x[n-order]]
            
        Returns:
            Tuple of (i, j, value) for selected pair
        """
        pairwise_values = []
        pairwise_indices = []
        
        # Build all pairwise combinations (including self-pairs)
        for i in range(self.order):
            for j in range(i, self.order):  # i <= j to avoid duplicates
                pair_value = (x_past[i] + x_past[j]) / 2.0
                pairwise_values.append(pair_value)
                pairwise_indices.append((i, j))
        
        # Select representative value
        if self.use_median:
            representative = np.median(pairwise_values)
        else:
            representative = np.mean(pairwise_values)
        
        # Find pair closest to representative value
        closest_idx = np.argmin(np.abs(pairwise_values - representative))
        i, j = pairwise_indices[closest_idx]
        
        return i, j, pairwise_values[closest_idx]
        
    def process_sample(self, x_current: float, x_past: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Process a single sample using pairwise median method.
        
        Args:
            x_current: Current sample x[n]
            x_past: Past samples [x[n-1], x[n-2], ..., x[n-order]]
            
        Returns:
            error: Prediction error e[n]
            coeffs: Updated coefficients
        """
        # Prediction
        prediction = np.dot(self.coeffs, x_past)
        error = x_current - prediction
        
        # Select pair using pairwise method
        i, j, _ = self._select_pair(x_past)
        
        # Update coefficients
        if i == j:
            # Single coefficient update (similar to original)
            self.coeffs[i] += self.mu * np.sign(error) * np.sign(x_past[i])
        else:
            # Weighted update of both coefficients
            total = abs(x_past[i]) + abs(x_past[j])
            if total > 0:
                weight_i = abs(x_past[i]) / total
                weight_j = 1 - weight_i
                
                self.coeffs[i] += self.mu * weight_i * np.sign(error) * np.sign(x_past[i])
                self.coeffs[j] += self.mu * weight_j * np.sign(error) * np.sign(x_past[j])
        
        return error, self.coeffs.copy()
    
    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process entire signal.
        
        Args:
            signal: Input signal
            
        Returns:
            compressed: Prediction errors
            coeff_history: History of coefficient updates
        """
        n = len(signal)
        compressed = np.zeros(n - self.order, dtype=np.float64)
        coeff_history = []
        
        for i in range(self.order, n):
            x_current = signal[i]
            x_past = signal[i-self.order:i][::-1]
            
            error, coeffs = self.process_sample(x_current, x_past)
            compressed[i - self.order] = error
            coeff_history.append(coeffs.copy())
            
        return compressed, np.array(coeff_history)
    
    @property
    def coefficients(self) -> np.ndarray:
        """Get current coefficients."""
        return self.coeffs.copy()


def compare_algorithms():
    """Compare original vs improved algorithm."""
    import matplotlib.pyplot as plt
    
    # Generate test signal
    np.random.seed(42)
    n_samples = 1000
    signal = np.random.randn(n_samples)
    for i in range(4, n_samples):
        signal[i] += 0.8*signal[i-1] - 0.5*signal[i-2] + 0.3*signal[i-3] - 0.2*signal[i-4]
    
    # Add outliers
    signal[200:210] += 10.0 * np.random.randn(10)
    signal[500] += 20.0
    signal[700:705] += 15.0 * np.random.randn(5)
    
    # Test both algorithms
    from core_2006 import L1AdaptiveFilter2006
    
    original_filter = L1AdaptiveFilter2006(order=4, mu=0.03)
    improved_filter = PairwiseMedianL1Filter(order=4, mu=0.03, use_median=True)
    
    original_result, original_coeffs = original_filter.process(signal)
    improved_result, improved_coeffs = improved_filter.process(signal)
    
    # Statistics
    print("="*70)
    print("СРАВНЕНИЕ: ОРИГИНАЛЬНЫЙ L1 vs УЛУЧШЕННЫЙ (ПОПАРНЫЕ ПОЛУСУММЫ)")
    print("="*70)
    
    orig_ratio = np.var(original_result) / np.var(signal[4:])
    impr_ratio = np.var(improved_result) / np.var(signal[4:])
    
    print(f"Оригинальный L1 (2006):")
    print(f"  Коэффициент сжатия: {orig_ratio:.4f}")
    print(f"  Макс ошибка: {np.max(np.abs(original_result)):.2f}")
    
    print(f"\nУлучшенный L1 (попарные полусуммы):")
    print(f"  Коэффициент сжатия: {impr_ratio:.4f}")
    print(f"  Макс ошибка: {np.max(np.abs(improved_result)):.2f}")
    
    improvement = (orig_ratio - impr_ratio) / orig_ratio * 100
    print(f"\nУЛУЧШЕНИЕ: {improvement:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coefficient evolution
    for i in range(4):
        axes[0, 0].plot(original_coeffs[:, i], alpha=0.7, label=f'Ориг. a{i}')
    axes[0, 0].set_title('Оригинальный L1: эволюция коэффициентов')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for i in range(4):
        axes[0, 1].plot(improved_coeffs[:, i], alpha=0.7, label=f'Улучш. a{i}')
    axes[0, 1].set_title('Улучшенный L1: эволюция коэффициентов')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error comparison
    window = 50
    def moving_avg(x, w):
        return np.convolve(x, np.ones(w)/w, mode='valid')
    
    axes[1, 0].plot(moving_avg(np.abs(original_result), window), 
                   'r-', alpha=0.7, label='Оригинальный')
    axes[1, 0].plot(moving_avg(np.abs(improved_result), window),
                   'g-', alpha=0.7, label='Улучшенный')
    axes[1, 0].set_title('Скользящее среднее |ошибки|')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of errors
    axes[1, 1].hist(original_result, bins=50, alpha=0.5, color='red', 
                   label='Оригинальный', density=True)
    axes[1, 1].hist(improved_result, bins=50, alpha=0.5, color='green',
                   label='Улучшенный', density=True)
    axes[1, 1].set_title('Распределение ошибок')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Сравнение: L1 оригинал (2006) vs L1 улучшенный (попарные полусуммы)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_pairwise.png', dpi=150, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    compare_algorithms()
