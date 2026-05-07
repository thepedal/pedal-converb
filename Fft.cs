// Fft.cs — fixed-size 512-point complex Cooley-Tukey FFT/IFFT.
//
// Scalar implementation by design: in the partitioned-convolution use
// case the FFT runs once per audio buffer per channel (4 calls total
// per Work), whereas the complex multiply-accumulate runs K times per
// buffer per channel (K = number of IR partitions, e.g. 1875 for 10 s
// of reverb at 48 kHz). For reverb-priority builds the MAC dominates
// total cost by 1-2 orders of magnitude, so SIMD effort is concentrated
// there (see ConvolutionEngine.ComplexMacAdd). At ~17 Mflops/s for the
// FFTs alone (4 × 187 buffers/s × 23 kflops) the scalar FFT is well
// inside budget.
//
// Twiddle and bit-reversal tables are precomputed once at type init
// and shared across all engines (immutable after construction, so
// safe to share — see Core §22 for the static-mutability footgun this
// avoids).

using System;

namespace PedalConVerb;

internal static class Fft
{
    public const int N = 512;          // FFT size (= 2 × hop = 2 × 256)
    private const int LOG2N = 9;       // log2(512)

    private static readonly int[]   _bitrev = new int[N];
    private static readonly float[] _twReal = new float[N / 2];
    private static readonly float[] _twImag = new float[N / 2];

    static Fft()
    {
        // Bit-reversal permutation indices.
        for (int i = 0; i < N; i++)
        {
            int rev = 0, x = i;
            for (int b = 0; b < LOG2N; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
            _bitrev[i] = rev;
        }
        // Forward twiddle factors: exp(-i·2π·k/N), k = 0..N/2-1.
        // Inverse FFT negates the imaginary part on the fly.
        for (int i = 0; i < N / 2; i++)
        {
            float angle = -2f * MathF.PI * i / N;
            _twReal[i] = MathF.Cos(angle);
            _twImag[i] = MathF.Sin(angle);
        }
    }

    /// <summary>In-place forward FFT. Arrays must be length N.</summary>
    public static void Forward(float[] real, float[] imag)
    {
        Permute(real, imag);
        Butterflies(real, imag, inverse: false);
    }

    /// <summary>In-place inverse FFT (with 1/N scaling). Arrays must be length N.</summary>
    public static void Inverse(float[] real, float[] imag)
    {
        Permute(real, imag);
        Butterflies(real, imag, inverse: true);
        const float inv = 1f / N;
        for (int i = 0; i < N; i++) { real[i] *= inv; imag[i] *= inv; }
    }

    private static void Permute(float[] real, float[] imag)
    {
        for (int i = 0; i < N; i++)
        {
            int j = _bitrev[i];
            if (j > i)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }
    }

    private static void Butterflies(float[] real, float[] imag, bool inverse)
    {
        // Decimation-in-time (DIT) Cooley-Tukey. Stage 'size' doubles each
        // pass; tableStep walks the precomputed twiddles at the right rate.
        for (int size = 2; size <= N; size <<= 1)
        {
            int halfSize  = size >> 1;
            int tableStep = N / size;
            for (int i = 0; i < N; i += size)
            {
                int k = 0;
                for (int j = i; j < i + halfSize; j++)
                {
                    float twRe = _twReal[k];
                    float twIm = inverse ? -_twImag[k] : _twImag[k];
                    int   u    = j + halfSize;

                    // t = X[u] · twiddle
                    float tre = real[u] * twRe - imag[u] * twIm;
                    float tim = real[u] * twIm + imag[u] * twRe;

                    // X[u] = X[j] - t ;  X[j] = X[j] + t
                    real[u] = real[j] - tre;
                    imag[u] = imag[j] - tim;
                    real[j] += tre;
                    imag[j] += tim;

                    k += tableStep;
                }
            }
        }
    }
}
