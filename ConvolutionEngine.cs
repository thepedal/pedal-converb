// ConvolutionEngine.cs — overlap-save partitioned frequency-domain
// convolution + SIMD complex multiply-accumulate.
//
// Algorithm (partition size N = 256, FFT size M = 2N = 512):
//
//   For each input block of N samples:
//     1. Build a 2N-sample frame: [previous block, current block]
//     2. FFT the frame  →  F_t  (push to FDL ring buffer at FdlHead)
//     3. Y_freq = Σ_k  FDL[(FdlHead − k) mod K] · IR_partition_freq[k]
//        (complex MAC across all K partitions; SIMD hot path)
//     4. IFFT(Y_freq) → 2N samples
//     5. Output = last N samples of IFFT result (overlap-save)
//
// The MAC dominates total CPU cost by orders of magnitude for reverb-
// length IRs (10 s ≈ 1875 partitions). Each partition contributes
// 512 complex multiply-adds per buffer; that's ~1 M cmul-adds per
// buffer. Vector<float> processes 8 (AVX2) or 4 (SSE) lanes at once,
// giving 4–8× speedup on the inner loop. The structure-of-arrays
// layout (separate real[] and imag[]) is what makes that vectorise
// cleanly — interleaved complex would cost shuffles.
//
// Stream-state (input accumulator, output queue, prev block, FDL head)
// is per-engine and persists across IR changes. The IR partition data
// and FDL buffers are owned by ConvolutionState (rebuilt + atomically
// swapped on the UI thread by the host machine; see Core §24).

using System;
using System.Numerics;

namespace PedalConVerb;

/// <summary>
/// FFT-domain representation of an IR, partitioned into N-sample blocks
/// each zero-padded to 2N and FFT'd. Real and Imag are stored as separate
/// arrays for SIMD-friendly access (Structure-of-Arrays).
/// </summary>
internal sealed class IrPartitionTable
{
    public int       NumPartitions;
    public float[][] Real;   // [K][2N]
    public float[][] Imag;   // [K][2N]
}

/// <summary>
/// Bundles per-direction IR data + FDL buffers. Built fully on the UI
/// thread, published atomically via volatile reference write — the audio
/// thread only ever reads the IR fields and reads/writes the FDL fields
/// it owns. Per-channel FDLs are independent buffers (input differs L/R)
/// even when the IrL/IrR refs alias for mono IRs.
/// </summary>
internal sealed class ConvolutionState
{
    public IrPartitionTable IrL;
    public IrPartitionTable IrR;
    public float[][]        FdlLReal;   // [K][2N]
    public float[][]        FdlLImag;
    public float[][]        FdlRReal;
    public float[][]        FdlRImag;
}

internal sealed class ConvolutionEngine
{
    public const int Hop   = 256;          // partition / hop size (N)
    public const int Frame = 2 * Hop;      // FFT frame size (2N = 512)

    // ── Persistent stream state (audio thread, owned by this engine) ─────
    private readonly float[] _inAcc     = new float[Hop];     // N input samples being accumulated
    private readonly float[] _outQ      = new float[Hop];     // N output samples ready to read
    private readonly float[] _prevBlock = new float[Hop];     // last N inputs (forms previous half of frame)
    private readonly float[] _frameRe   = new float[Frame];   // scratch for FFT input
    private readonly float[] _frameIm   = new float[Frame];
    private readonly float[] _outRe     = new float[Frame];   // scratch for IFFT input/output
    private readonly float[] _outIm     = new float[Frame];

    // _pos walks 0..Hop-1 in lockstep for both input fill and output drain
    // (one input sample per output sample), so we only need one counter.
    private int _pos;
    private int _fdlHead;

    // ── Externally-owned state (set by host on IR change) ────────────────
    private IrPartitionTable _ir;
    private float[][]        _fdlReal;
    private float[][]        _fdlImag;

    /// <summary>
    /// Set the IR + FDL buffers for this engine. If the IR ref is identical
    /// to the current one this is a no-op. On change, all stream state is
    /// reset (brief glitch; FDL starts from zeros).
    /// </summary>
    public void SetState(IrPartitionTable ir, float[][] fdlReal, float[][] fdlImag)
    {
        if (ReferenceEquals(_ir, ir) && ReferenceEquals(_fdlReal, fdlReal)) return;
        _ir       = ir;
        _fdlReal  = fdlReal;
        _fdlImag  = fdlImag;
        _fdlHead  = 0;
        _pos      = 0;
        Array.Clear(_inAcc,     0, Hop);
        Array.Clear(_outQ,      0, Hop);
        Array.Clear(_prevBlock, 0, Hop);
        // FDL buffers have just been allocated fresh by the host; already zero.
    }

    /// <summary>
    /// Process n samples from input → output (each ±1.0 normalised float).
    /// When the IR is null, fills output with zeros (host mixes dry separately).
    /// Latency = Hop samples (one block).
    /// </summary>
    public void Process(float[] input, int inOff, float[] output, int outOff, int n)
    {
        if (_ir == null || _ir.NumPartitions == 0)
        {
            Array.Clear(output, outOff, n);
            return;
        }

        int processed = 0;
        while (processed < n)
        {
            int chunk = Math.Min(Hop - _pos, n - processed);

            // Drain output queue and fill input accumulator simultaneously.
            // Pos is shared because input rate == output rate.
            Array.Copy(_outQ,                       _pos, output,  outOff + processed, chunk);
            Array.Copy(input, inOff + processed,    _inAcc, _pos,                       chunk);
            _pos       += chunk;
            processed  += chunk;

            if (_pos == Hop)
            {
                ProcessBlock();
                _pos = 0;
            }
        }
    }

    private void ProcessBlock()
    {
        // ── Step 1: build 2N frame [prev, current], FFT it ───────────────
        Array.Copy(_prevBlock, 0, _frameRe, 0,   Hop);
        Array.Copy(_inAcc,     0, _frameRe, Hop, Hop);
        Array.Clear(_frameIm, 0, Frame);
        Fft.Forward(_frameRe, _frameIm);

        // ── Step 2: push F_t to FDL at the new head ──────────────────────
        int K = _ir.NumPartitions;
        _fdlHead = (_fdlHead + 1) % K;
        Array.Copy(_frameRe, 0, _fdlReal[_fdlHead], 0, Frame);
        Array.Copy(_frameIm, 0, _fdlImag[_fdlHead], 0, Frame);

        // ── Step 3: complex MAC across all partitions ────────────────────
        // Y_freq = Σ_k  FDL[(head−k) mod K] · IR_freq[k]
        //
        // For K = 0 we returned earlier; here K ≥ 1.
        // The first partition (k=0) writes (no accumulate) so we can skip
        // a pre-clear pass over OutRe/OutIm — saves 2 × 2N = 4096 stores
        // per buffer.
        ComplexMul(_fdlReal[_fdlHead], _fdlImag[_fdlHead],
                   _ir.Real[0],        _ir.Imag[0],
                   _outRe,             _outIm);

        for (int k = 1; k < K; k++)
        {
            int idx = _fdlHead - k;
            if (idx < 0) idx += K;
            ComplexMacAdd(_fdlReal[idx], _fdlImag[idx],
                          _ir.Real[k],   _ir.Imag[k],
                          _outRe,        _outIm);
        }

        // ── Step 4: inverse FFT ──────────────────────────────────────────
        Fft.Inverse(_outRe, _outIm);

        // ── Step 5: output = last N samples of IFFT result ───────────────
        // (First N samples are the wrap-around portion of circular conv;
        //  the last N are the valid linear conv output for this block.)
        Array.Copy(_outRe, Hop, _outQ, 0, Hop);

        // Save current input block as "previous" for the next frame.
        Array.Copy(_inAcc, 0, _prevBlock, 0, Hop);
    }

    // ── SIMD complex multiply-accumulate (audio-thread hot path) ─────────
    //
    // Both forms below process Frame = 512 floats per call. Frame is
    // divisible by Vector<float>.Count for any plausible width (4 SSE,
    // 8 AVX2, 16 AVX-512), so the scalar tail is empty in practice — but
    // it's kept for portability if a future runtime exposes width 6 etc.
    //
    // Complex multiply in SoA form:
    //   (a + bi)·(c + di) = (ac − bd) + (ad + bc)i

    /// <summary>y_freq = x_freq · h_freq  (no accumulate; first partition).</summary>
    private static void ComplexMul(float[] xr, float[] xi,
                                   float[] hr, float[] hi,
                                   float[] yr, float[] yi)
    {
        int n     = Frame;
        int width = Vector<float>.Count;
        int simd  = n - (n % width);
        int i     = 0;

        for (; i < simd; i += width)
        {
            var a = new Vector<float>(xr, i);
            var b = new Vector<float>(xi, i);
            var c = new Vector<float>(hr, i);
            var d = new Vector<float>(hi, i);
            (a * c - b * d).CopyTo(yr, i);
            (a * d + b * c).CopyTo(yi, i);
        }
        for (; i < n; i++)
        {
            yr[i] = xr[i] * hr[i] - xi[i] * hi[i];
            yi[i] = xr[i] * hi[i] + xi[i] * hr[i];
        }
    }

    /// <summary>y_freq += x_freq · h_freq  (the inner loop's hot kernel).</summary>
    private static void ComplexMacAdd(float[] xr, float[] xi,
                                      float[] hr, float[] hi,
                                      float[] yr, float[] yi)
    {
        int n     = Frame;
        int width = Vector<float>.Count;
        int simd  = n - (n % width);
        int i     = 0;

        for (; i < simd; i += width)
        {
            var a    = new Vector<float>(xr, i);
            var b    = new Vector<float>(xi, i);
            var c    = new Vector<float>(hr, i);
            var d    = new Vector<float>(hi, i);
            var yrV  = new Vector<float>(yr, i);
            var yiV  = new Vector<float>(yi, i);
            (yrV + a * c - b * d).CopyTo(yr, i);
            (yiV + a * d + b * c).CopyTo(yi, i);
        }
        for (; i < n; i++)
        {
            yr[i] += xr[i] * hr[i] - xi[i] * hi[i];
            yi[i] += xr[i] * hi[i] + xi[i] * hr[i];
        }
    }
}
