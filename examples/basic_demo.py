"""
Basic demonstration of the 2006 L1-adaptive filter.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.core_2006 import L1AdaptiveFilter2006, create_test_signal


def main():
    """Run basic demonstration."""
    print("=" * 70)
    print("L1-RENAISSANCE: 2006 L1-ADAPTIVE FILTER DEMONSTRATION")
    print("=" * 70)
    
    # 1. Generate test signal (AR process)
    print("\n1. Generating test signal (AR process)...")
    signal = create_test_signal(length=1000)
    print(f"   Signal length: {len(signal)} samples")
    print(f"   Signal variance: {np.var(signal):.4f}")
    
    # 2. Initialize filter with original 2006 parameters
    print("\n2. Initializing L1-adaptive filter...")
    filter = L1AdaptiveFilter2006(order=4, mu=0.03, bt=16.0)
    print(f"   Filter order: {filter.order}")
    print(f"   Learning rate (μ): {filter.mu}")
    print(f"   Parameter bt: {filter.bt}")
    print(f"   Initial coefficients: {filter.coefficients}")
    
    # 3. Process signal
    print("\n3. Processing signal...")
    compressed, coeff_history = filter.process(signal)
    
    print(f"   Compressed signal length: {len(compressed)}")
    print(f"   Compressed signal variance: {np.var(compressed):.4f}")
    
    # 4. Calculate compression ratio
    ratio = filter.get_compression_ratio(signal)
    print(f"\n4. Compression ratio (variance reduction):")
    print(f"   Original variance: {np.var(signal[4:]):.4f}")
    print(f"   Compressed variance: {np.var(compressed):.4f}")
    print(f"   Ratio (compressed/original): {ratio:.4f}")
    print(f"   Information: {'Good compression' if ratio < 1 else 'Limited compression'}")
    
    # 5. Test lossless reconstruction
    print("\n5. Testing lossless reconstruction...")
    initial_samples = signal[:4]
    reconstructed = filter.decompress(compressed, initial_samples)
    
    reconstruction_error = np.max(np.abs(signal - reconstructed))
    print(f"   Maximum reconstruction error: {reconstruction_error:.2e}")
    print(f"   Reconstruction: {'SUCCESS' if reconstruction_error < 1e-10 else 'FAILED'}")
    
    # 6. Visualize results
    print("\n6. Generating visualizations...")
    visualize_results(signal, compressed, coeff_history, filter.coefficients)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)


def visualize_results(signal, compressed, coeff_history, final_coeffs):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original vs compressed signal (first 100 samples)
    axes[0, 0].plot(signal[4:104], 'b-', alpha=0.7, label='Original')
    axes[0, 0].plot(compressed[:100], 'r-', alpha=0.7, label='Compressed')
    axes[0, 0].set_xlabel('Sample index')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Original vs Compressed Signal (first 100 samples)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Coefficient evolution
    for i in range(4):
        axes[0, 1].plot(coeff_history[:, i], label=f'Coefficient {i}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Coefficient Evolution During Adaptation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of prediction errors
    axes[0, 2].hist(compressed, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Prediction Error')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Prediction Errors')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Final coefficients
    bars = axes[1, 0].bar(range(len(final_coeffs)), final_coeffs)
    axes[1, 0].set_xlabel('Coefficient Index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Final Filter Coefficients')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, value in zip(bars, final_coeffs):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 5. Power spectral density
    from scipy import signal as sp_signal
    f_orig, Pxx_orig = sp_signal.welch(signal[4:], fs=8000, nperseg=256)
    f_comp, Pxx_comp = sp_signal.welch(compressed, fs=8000, nperseg=256)
    
    axes[1, 1].semilogy(f_orig, Pxx_orig, 'b-', alpha=0.7, label='Original')
    axes[1, 1].semilogy(f_comp, Pxx_comp, 'r-', alpha=0.7, label='Compressed')
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('PSD [V²/Hz]')
    axes[1, 1].set_title('Power Spectral Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Scatter plot: prediction error vs signal amplitude
    axes[1, 2].scatter(signal[4:4+len(compressed)], compressed, alpha=0.5, s=1)
    axes[1, 2].set_xlabel('Original Signal Amplitude')
    axes[1, 2].set_ylabel('Prediction Error')
    axes[1, 2].set_title('Prediction Error vs Signal Amplitude')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('l1_adaptive_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("   Visualizations saved to 'l1_adaptive_demo.png'")


if __name__ == "__main__":
    main()
