# Pedal ConVerb

A managed convolution reverb / cab-sim for ReBuzz. Loads an impulse response
from any wave-table slot and convolves the input through it in real time
using uniformly-partitioned overlap-save FFT convolution with a SIMD complex
multiply-accumulate hot path.

Reverb-priority: tuned for medium-to-long IRs (rooms, halls, plates) where
the per-block latency cost is dwarfed by the IR length. For short IRs (cab
sims, transient shapers) it works fine but isn't the lowest-latency design
possible — see *Future work* below.

## Parameters

| Param   | Range  | Default | Notes                                             |
|---------|--------|---------|---------------------------------------------------|
| IR Wave | 0..200 | 0       | Wave-table slot. 0 = bypass (dry only).           |
| Dry     | 0..127 | 0       | Dry signal level.                                 |
| Wet     | 0..127 | 127     | Wet (convolved) signal level. 127 = unity.        |
| Output  | 0..48  | 24      | Output trim, ±24 dB in 1-dB steps. 24 = 0 dB.     |

The Output range is stored as a non-negative byte with an explicit −24
offset applied in the DSP path — this avoids the negative-range parameter
offset bug noted in PedalComp §2.

To use as a typical insert reverb: leave Wet at 127, dial Dry to taste.
To use as a 100%-wet send return: Dry = 0, Wet = 127.
To use as a cab sim insert: Dry = 0, Wet = 127, pick a cab IR and adjust
Output for unity-loudness match.

## Design

### Convolution
- Uniform 256-sample partitions (`Hop = 256`), 512-point FFT frame
  (`Frame = 2 × Hop`).
- Overlap-save: each block's frame is `[previous 256, current 256]`,
  FFT'd, MAC'd against all `K` IR partitions in the frequency-domain
  delay line, IFFT'd; the last 256 samples form the output for that
  block.
- One forward FFT per channel per buffer; one inverse FFT per channel
  per buffer; `K` complex multiply-accumulates per channel per buffer
  (`K = ⌈IR_length / 256⌉`).

### SIMD MAC
- Frequency-domain data is stored Structure-of-Arrays (separate `Real[]`
  and `Imag[]` arrays per partition / FDL slot). This is what makes the
  inner kernel vectorise without shuffles.
- `ConvolutionEngine.ComplexMacAdd` uses `System.Numerics.Vector<float>`,
  which auto-targets 4-wide SSE or 8-wide AVX2 depending on the host CPU.
  `Frame = 512` is divisible by every plausible vector width (4, 8, 16),
  so the scalar tail is empty in practice.
- The first partition uses a non-accumulating multiply (`ComplexMul`),
  saving 4096 stores per buffer that would otherwise be needed to
  pre-clear the output frame.

### Stereo handling (auto)
- Mono IR → both channels share one `IrPartitionTable` (separate FDLs;
  L/R inputs differ).
- Stereo IR → channel 0 → IrL, channel 1 → IrR; independent partition
  tables, no cross-coupling.

### Threading
- IR rebuild happens on the UI thread, triggered by:
  - Polling timer (200 ms) detecting an `IRWave` parameter change.
  - `IWave.PropertyChanged` on the subscribed wave (e.g. user reloads
    the WAV in the same slot).
- Built `ConvolutionState` is published to the audio thread via a
  single `volatile` reference write (Core §24 atomic publish pattern).
- Audio thread reads the state once per `Work()` and uses it directly.
  The IR partition arrays are immutable post-publish; the FDL arrays
  are mutated only by the audio thread and only those it owns at the
  current state ref.

## Limitations / future work

- **Latency.** Fixed ~5.3 ms at 48 kHz (one `Hop` of latency). For lower-
  latency cab-sim use, a non-uniform partition scheme (smaller first
  partition + larger tail partitions) would give ~0.3 ms latency at
  similar CPU cost. Not implemented in v1.
- **IR cap.** 30 s @ 48 kHz (1 440 000 samples). Longer waves are
  silently truncated with a debug-console note.
- **No sample-rate conversion.** The IR is used at the engine sample
  rate regardless of its recorded rate. Mismatch shifts the IR's
  spectral content. Workaround: pre-resample the WAV externally.
- **UI-thread IR build.** A 30 s IR rebuilds in 30–60 ms on the UI
  thread, which can hitch the GUI. Could be moved to a background task
  in v2.
- **Conjugate-symmetry redundancy.** The 2N-point complex FFT stores
  full 2N-bin spectra in the FDL when only N+1 are unique. Halving this
  via real-FFT packing would cut MAC work by ~50 % at the cost of
  significantly more complex bookkeeping.
- **Click on IR change.** The FDL is reset when the IR ref changes,
  causing a brief discontinuity. Fading the wet path during the swap
  would be smoother.

## Files

- `PedalConVerb.csproj` — build hygiene, references, post-build deploy.
- `PedalConVerb.cs` — `IBuzzMachine` class, parameters, threading, mix.
- `ConvolutionEngine.cs` — overlap-save engine + SIMD complex MAC.
- `Fft.cs` — 512-point Cooley-Tukey FFT/IFFT.
- `IrPrep.cs` — wavetable layer → partitioned frequency-domain IR.

## Build

```
dotnet build -c Release
```

Built DLL is named `Pedal ConVerb.NET.dll` (the `.NET` suffix is
mandatory — Build §2) and is auto-copied to
`C:\Program Files\ReBuzz\Gear\Effects\` by the post-build target.
