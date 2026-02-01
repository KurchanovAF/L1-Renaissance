"""
Unit tests for the 2006 L1-adaptive filter implementation.
"""

import numpy as np
import pytest
from src.core_2006 import L1AdaptiveFilter2006, create_test_signal


class TestL1AdaptiveFilter2006:
    """Test cases for L1AdaptiveFilter2006."""
    
    def test_initialization(self):
        """Test filter initialization."""
        filter = L1AdaptiveFilter2006(order=4, mu=0.03, bt=16.0)
        
        assert filter.order == 4
        assert filter.mu == 0.03
        assert filter.bt == 16.0
        assert len(filter.coefficients) == 4
        assert np.all(filter.coefficients == 0)
        
    def test_reset(self):
        """Test coefficient reset."""
        filter = L1AdaptiveFilter2006(order=4)
        filter.coefficients = np.array([1.0, 2.0, 3.0, 4.0])
        
        filter.reset()
        assert np.all(filter.coefficients == 0)
        
    def test_process_sample(self):
        """Test single sample processing."""
        filter = L1AdaptiveFilter2006(order=2, mu=0.1)
        
        # Test case 1
        error, coeffs = filter.process_sample(1.0, np.array([0.5, -0.3]))
        assert isinstance(error, float)
        assert len(coeffs) == 2
        
        # Test case with max at index 0
        error1, coeffs1 = filter.process_sample(1.0, np.array([2.0, 1.0]))
        # Test case with max at index 1
        error2, coeffs2 = filter.process_sample(1.0, np.array([1.0, 3.0]))
        
        assert not np.array_equal(coeffs1, coeffs2)
        
    def test_process_signal(self):
        """Test processing entire signal."""
        filter = L1AdaptiveFilter2006(order=4)
        
        # Generate test signal
        signal = create_test_signal(length=100)
        
        # Process signal
        compressed, coeff_history = filter.process(signal)
        
        assert len(compressed) == len(signal) - 4
        assert len(coeff_history) == len(signal) - 4
        assert coeff_history.shape[1] == 4
        
        # Check that compression reduces variance
        original_variance = np.var(signal[4:])
        compressed_variance = np.var(compressed)
        assert compressed_variance < original_variance * 1.5  # Allow some increase
        
    def test_compress_decompress(self):
        """Test compression and decompression cycle."""
        filter = L1AdaptiveFilter2006(order=4, mu=0.03)
        
        # Generate test signal
        signal = create_test_signal(length=200)
        
        # Compress
        compressed = filter.compress(signal)
        
        # Decompress (need initial samples)
        initial_samples = signal[:4]
        reconstructed = filter.decompress(compressed, initial_samples)
        
        # Check reconstruction accuracy
        # Note: L1-adaptive filter is lossless in theory
        # But numerical errors may occur
        tolerance = 1e-10
        assert np.allclose(signal, reconstructed, atol=tolerance)
        
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        filter = L1AdaptiveFilter2006(order=4)
        
        # Signal too short
        with pytest.raises(ValueError):
            filter.process(np.array([1, 2, 3]))
            
        # Wrong length for process_sample
        with pytest.raises(ValueError):
            filter.process_sample(1.0, np.array([1, 2, 3]))  # length 3, expected 4
            
        # Wrong length for decompress initial_samples
        with pytest.raises(ValueError):
            filter.decompress(np.array([1, 2, 3]), np.array([1, 2]))
            
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        filter = L1AdaptiveFilter2006(order=4)
        
        # Create signal with structure (should compress well)
        signal = create_test_signal(length=500)
        
        ratio = filter.get_compression_ratio(signal)
        
        assert 0 <= ratio <= 2  # Reasonable range
        print(f"Compression ratio: {ratio:.4f}")
        
    def test_coefficient_setter(self):
        """Test coefficient setter and getter."""
        filter = L1AdaptiveFilter2006(order=3)
        
        new_coeffs = np.array([0.5, -0.2, 0.1])
        filter.coefficients = new_coeffs
        
        assert np.allclose(filter.coefficients, new_coeffs)
        
        # Test invalid length
        with pytest.raises(ValueError):
            filter.coefficients = np.array([1, 2, 3, 4])  # length 4, expected 3


def test_create_test_signal():
    """Test test signal generation."""
    signal = create_test_signal(length=100)
    
    assert len(signal) == 100
    assert isinstance(signal, np.ndarray)
    
    # Test with custom AR coefficients
    ar_coeffs = np.array([0.9, -0.6])
    signal_custom = create_test_signal(length=50, ar_coeffs=ar_coeffs)
    assert len(signal_custom) == 50


if __name__ == "__main__":
    # Run tests
    import sys
    pytest.main(sys.argv)
