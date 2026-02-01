"""
Exact implementation of the 2006 L1-adaptive filter algorithm.

Based on: "О применении метрики L1 при вычислении коэффициентов 
линейного предсказания для сжатия речевого сигнала"
Author: Alexey Kurchanov, May 29, 2006
"""

import numpy as np
from typing import Tuple, Optional


class L1AdaptiveFilter2006:
    """
    Original 2006 L1-adaptive filter implementation.
    
    Key features:
    - Uses L1 metric instead of traditional L2
    - Updates only one coefficient per sample (greedy approach)
    - Extremely computationally efficient
    - Robust to outliers
    
    Parameters as in original paper:
    - order: filter order (default 4)
    - mu: learning rate (default 0.03)
    - bt: scaling parameter (default 16.0)
    """
    
    def __init__(self, order: int = 4, mu: float = 0.03, bt: float = 16.0):
        """
        Initialize the L1-adaptive filter.
        
        Args:
            order: Filter order (number of coefficients)
            mu: Learning rate (step size)
            bt: Parameter from original paper
        """
        self.order = order
        self.mu = mu
        self.bt = bt
        self.coeffs = np.zeros(order, dtype=np.float64)
        self.reset()
        
    def reset(self) -> None:
        """Reset filter coefficients to zero."""
        self.coeffs = np.zeros(self.order, dtype=np.float64)
        
    def process_sample(self, x_current: float, x_past: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Process a single sample using the original 2006 algorithm.
        
        Args:
            x_current: Current sample x[n]
            x_past: Array of past samples [x[n-1], x[n-2], ..., x[n-order]]
            
        Returns:
            error: Prediction error e[n]
            coeffs: Updated coefficients
        """
        # Input validation
        if len(x_past) != self.order:
            raise ValueError(f"x_past must have length {self.order}, got {len(x_past)}")
            
        # 1. Linear prediction
        prediction = np.dot(self.coeffs, x_past)
        error = x_current - prediction
        
        # 2. Find coefficient to update (original 2006 rule)
        #    Update only the coefficient corresponding to max(|x_past|)
        abs_x_past = np.abs(x_past)
        max_idx = np.argmax(abs_x_past)
        
        # 3. Update rule from 2006 paper
        #    a_k_new = a_k_old + μ * sign(e) * sign(x[n-k-1])
        if abs_x_past[max_idx] > 0:  # Avoid division by zero
            self.coeffs[max_idx] += self.mu * np.sign(error) * np.sign(x_past[max_idx])
        
        return error, self.coeffs.copy()
    
    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process entire signal using the 2006 algorithm.
        
        Args:
            signal: 1D numpy array of audio samples
            
        Returns:
            compressed: Prediction errors (compressed signal)
            coeff_history: History of coefficient updates
        """
        n = len(signal)
        if n <= self.order:
            raise ValueError(f"Signal length ({n}) must be greater than filter order ({self.order})")
            
        compressed = np.zeros(n - self.order, dtype=np.float64)
        coeff_history = []
        
        for i in range(self.order, n):
            x_current = signal[i]
            x_past = signal[i-self.order:i][::-1]  # [x[n-1], x[n-2], ...]
            
            error, coeffs = self.process_sample(x_current, x_past)
            compressed[i - self.order] = error
            coeff_history.append(coeffs.copy())
            
        return compressed, np.array(coeff_history)
    
    def compress(self, signal: np.ndarray) -> np.ndarray:
        """
        Compress signal (return only prediction errors).
        
        Args:
            signal: Input signal
            
        Returns:
            compressed: Prediction errors
        """
        compressed, _ = self.process(signal)
        return compressed
    
    def decompress(self, compressed: np.ndarray, initial_samples: np.ndarray) -> np.ndarray:
        """
        Decompress signal from prediction errors.
        
        Args:
            compressed: Prediction errors
            initial_samples: First 'order' samples of original signal
            
        Returns:
            reconstructed: Reconstructed signal
        """
        if len(initial_samples) != self.order:
            raise ValueError(f"initial_samples must have length {self.order}")
            
        n = len(compressed) + self.order
        reconstructed = np.zeros(n, dtype=np.float64)
        reconstructed[:self.order] = initial_samples
        
        # Reset coefficients for decompression
        original_coeffs = self.coeffs.copy()
        self.reset()
        
        for i in range(self.order, n):
            x_past = reconstructed[i-self.order:i][::-1]
            
            # Prediction using current coefficients
            prediction = np.dot(self.coeffs, x_past)
            
            # Reconstruct sample: x[n] = e[n] + prediction
            reconstructed[i] = compressed[i - self.order] + prediction
            
            # Update coefficients (same as compression)
            abs_x_past = np.abs(x_past)
            max_idx = np.argmax(abs_x_past)
            if abs_x_past[max_idx] > 0:
                error = compressed[i - self.order]
                self.coeffs[max_idx] += self.mu * np.sign(error) * np.sign(x_past[max_idx])
        
        # Restore original coefficients
        self.coeffs = original_coeffs
        
        return reconstructed
    
    def get_compression_ratio(self, original_signal: np.ndarray) -> float:
        """
        Calculate compression ratio (variance reduction).
        
        Args:
            original_signal: Original input signal
            
        Returns:
            ratio: variance(compressed) / variance(original)
        """
        compressed = self.compress(original_signal)
        original_variance = np.var(original_signal[self.order:])
        compressed_variance = np.var(compressed)
        
        if original_variance == 0:
            return 0.0
            
        return compressed_variance / original_variance
    
    @property
    def coefficients(self) -> np.ndarray:
        """Get current filter coefficients."""
        return self.coeffs.copy()
    
    @coefficients.setter
    def coefficients(self, coeffs: np.ndarray) -> None:
        """Set filter coefficients."""
        if len(coeffs) != self.order:
            raise ValueError(f"Coefficients must have length {self.order}")
        self.coeffs = np.array(coeffs, dtype=np.float64)


def create_test_signal(length: int = 1000, ar_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create test AR signal for demonstration.
    
    Args:
        length: Signal length
        ar_coeffs: AR coefficients (default: [0.8, -0.5, 0.3, -0.2])
        
    Returns:
        signal: Generated AR signal
    """
    if ar_coeffs is None:
        ar_coeffs = np.array([0.8, -0.5, 0.3, -0.2])
    
    order = len(ar_coeffs)
    signal = np.random.randn(length)
    
    # Add AR structure
    for i in range(order, length):
        for j in range(order):
            signal[i] += ar_coeffs[j] * signal[i - j - 1]
    
    return signal
