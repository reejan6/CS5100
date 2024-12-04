# Fast Fourier Transform (FFT)

The Fast Fourier Transform (FFT) is an efficient algorithm for computing the Discrete Fourier Transform (DFT) and its inverse. It is used to convert sequences of complex numbers between the time domain and the frequency domain.

## Discrete Fourier Transform (DFT)

The DFT of a sequence of \( N \) complex numbers \( x_0, x_1, \ldots, x_{N-1} \) is defined as:

\[
X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2 \pi k n / N}, \quad \text{for } k = 0, 1, \ldots, N-1
\]

Where:
- \( X_k \) represents the DFT coefficients.
- \( i \) is the imaginary unit (\( i^2 = -1 \)).
- \( e^{-i 2 \pi k n / N} \) is the complex exponential function.

## Inverse Discrete Fourier Transform (IDFT)

The IDFT is the inverse operation, converting frequency-domain data back into the time domain:

\[
x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{i 2 \pi k n / N}, \quad \text{for } n = 0, 1, \ldots, N-1
\]

## Fast Fourier Transform (FFT)

FFT significantly reduces the computational complexity of the DFT from \( O(N^2) \) to \( O(N \log N) \). This is achieved by breaking the DFT of a sequence into smaller, more manageable subproblems.

### Cooley-Tukey Algorithm

The Cooley-Tukey algorithm is the most widely used FFT method. It divides the DFT into two smaller DFTs: one for even-indexed elements and another for odd-indexed elements. 

1. **Divide**: Separate the sequence \( x \) into two halves:
   - Even-indexed elements: \( x_{\text{even}} = x_0, x_2, x_4, \ldots, x_{N-2} \)
   - Odd-indexed elements: \( x_{\text{odd}} = x_1, x_3, x_5, \ldots, x_{N-1} \)
2. **Conquer**: Recursively compute the DFT for each half.
3. **Combine**: Merge the results:

\[
X_k = X_k^{\text{even}} + e^{-i 2 \pi k / N} X_k^{\text{odd}}
\]
\[
X_{k+N/2} = X_k^{\text{even}} - e^{-i 2 \pi k / N} X_k^{\text{odd}}
\]

This is done for \( k = 0, 1, \ldots, N/2 - 1 \).

### Radix-2 FFT

Radix-2 FFT is a special case of the Cooley-Tukey algorithm where \( N \) is a power of 2. This simplifies the recursion and makes the computation highly efficient.

#### Butterfly Diagram

The combination process in Radix-2 FFT is visualized using a butterfly diagram, which depicts how values are merged and recombined in each step.

#### Example for \( N = 8 \)

1. Split the sequence into even and odd indexed elements.
2. Apply FFT recursively to each subset.
3. Combine the results using a butterfly structure.

## Applications

FFT is widely utilized across various domains:
- Signal processing
- Image processing
- Convolution operations
- Solving partial differential equations

## Conclusion

The FFT is a cornerstone of modern computational techniques, enabling fast and efficient DFT computation. Its recursive design and ability to combine results effectively make it an essential tool in numerous applications.

For more details, refer to textbooks on numerical methods or signal processing.

