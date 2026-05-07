// IrPrep.cs — UI-thread IR preparation.
//
// Reads the chosen wavetable layer, applies any post-load gain, slices
// the time-domain IR into N-sample partitions, FFTs each partition into
// the SoA frequency layout used by ConvolutionEngine, and pre-allocates
// the per-channel FDL buffers. The returned ConvolutionState is then
// published to the audio thread via a single volatile reference write.
//
// All allocation happens here, never on the audio thread (Core §16, §24).
//
// Stereo policy ("auto"):
//   • mono IR layer  → one IrPartitionTable shared by L and R engines
//                      (FDL buffers are still allocated separately because
//                      L and R input differ)
//   • stereo IR      → channel 0 → IrL, channel 1 → IrR
//                      (independent partition tables, no cross-coupling)
//
// IR length cap: 30 seconds at 48 kHz nominal (1 440 000 samples). This
// keeps memory bounded; a full 30 s stereo IR uses ~46 MB across two
// partition tables + two FDLs. Longer waves are silently truncated and
// a one-line note is written to the debug console.

using System;
using BuzzGUI.Interfaces;

namespace PedalConVerb;

internal static class IrPrep
{
    private const int MaxIrSamples = 30 * 48000;   // ~30 s at 48 kHz

    /// <summary>
    /// Build a fresh ConvolutionState from a wavetable IWave. Returns null
    /// when the wave is missing/empty/invalid — caller treats null as
    /// "bypass, dry only".
    /// </summary>
    public static ConvolutionState Build(IWave wave, float gainLin, IBuzz dcLog)
    {
        if (wave == null || wave.Layers == null || wave.Layers.Count == 0) return null;
        var layer = wave.Layers[0];
        if (layer == null) return null;

        int len = layer.SampleCount;
        if (len <= 0) return null;
        if (len > MaxIrSamples)
        {
            try { dcLog?.DCWriteLine($"[Pedal ConVerb] IR truncated from {len} to {MaxIrSamples} samples (~30 s @ 48 kHz cap)."); }
            catch { }
            len = MaxIrSamples;
        }

        bool isStereo = (wave.Flags & WaveFlags.Stereo) != 0;

        // ── Read time-domain IR data ─────────────────────────────────────
        // GetDataAsFloat normalises to ±1.0 internally regardless of the
        // wave's storage format (Int16/24/32, Float32). outstride=1, no
        // padding here — partition build below copies into a 2N FFT buffer
        // with implicit zero-padding.
        float[] dL, dR = null;
        try
        {
            dL = new float[len];
            layer.GetDataAsFloat(dL, 0, 1, 0, 0, len);
            if (isStereo)
            {
                dR = new float[len];
                layer.GetDataAsFloat(dR, 0, 1, 1, 0, len);
            }
        }
        catch { return null; }

        // Apply IR-build gain in-place. Cheap; runs once.
        if (gainLin != 1f)
        {
            for (int i = 0; i < len; i++) dL[i] *= gainLin;
            if (dR != null) for (int i = 0; i < len; i++) dR[i] *= gainLin;
        }

        var irL = BuildPartitionTable(dL, len);
        if (irL == null) return null;

        IrPartitionTable irR;
        if (dR != null)
        {
            irR = BuildPartitionTable(dR, len);
            if (irR == null) irR = irL;       // fall back if R partition build fails
        }
        else
        {
            irR = irL;                        // mono: share partition table
        }

        // ── Pre-allocate per-channel FDL buffers ─────────────────────────
        int K = irL.NumPartitions;
        var fdlLR = new float[K][];
        var fdlLI = new float[K][];
        var fdlRR = new float[K][];
        var fdlRI = new float[K][];
        for (int k = 0; k < K; k++)
        {
            fdlLR[k] = new float[ConvolutionEngine.Frame];
            fdlLI[k] = new float[ConvolutionEngine.Frame];
            fdlRR[k] = new float[ConvolutionEngine.Frame];
            fdlRI[k] = new float[ConvolutionEngine.Frame];
        }

        return new ConvolutionState
        {
            IrL      = irL,
            IrR      = irR,
            FdlLReal = fdlLR,
            FdlLImag = fdlLI,
            FdlRReal = fdlRR,
            FdlRImag = fdlRI,
        };
    }

    private static IrPartitionTable BuildPartitionTable(float[] ir, int len)
    {
        int hop  = ConvolutionEngine.Hop;     // 256
        int frm  = ConvolutionEngine.Frame;   // 512
        int K    = (len + hop - 1) / hop;
        if (K == 0) return null;

        var tbl = new IrPartitionTable
        {
            NumPartitions = K,
            Real = new float[K][],
            Imag = new float[K][],
        };

        // Re-used scratch for the FFT — partitions are built sequentially
        // on the UI thread, no concurrency concern.
        var sR = new float[frm];
        var sI = new float[frm];

        for (int k = 0; k < K; k++)
        {
            // Zero-pad partition k to 2N: [N IR samples, N zeros].
            // The trailing N zeros are what make 2N-point circular conv
            // equal to linear conv on the relevant output range.
            Array.Clear(sR, 0, frm);
            Array.Clear(sI, 0, frm);

            int copyLen = Math.Min(hop, len - k * hop);
            Array.Copy(ir, k * hop, sR, 0, copyLen);
            // sR[copyLen..frm-1] left zero by Array.Clear above.

            Fft.Forward(sR, sI);

            tbl.Real[k] = new float[frm];
            tbl.Imag[k] = new float[frm];
            Array.Copy(sR, tbl.Real[k], frm);
            Array.Copy(sI, tbl.Imag[k], frm);
        }

        return tbl;
    }
}
