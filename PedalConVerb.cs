// PedalConVerb.cs — top-level managed effect machine.
//
// Parameters:
//   • IR Wave  (byte, IsWaveNumber=true, 0..200, def 0)
//       Wave-table slot for the impulse response. 0 = bypass (dry only).
//   • Dry      (byte, 0..127, def 0)            — dry mix amount
//   • Wet      (byte, 0..127, def 127 = unity)  — wet (convolved) amount
//   • Output   (byte, 0..48,  def 24 = 0 dB)    — output trim, ±24 dB
//                                                  (offset stored to keep
//                                                   MinValue ≥ 0; see
//                                                   PedalComp §2)
//
// Threading:
//   • UI thread polls 4 params @200 ms via System.Threading.Timer →
//     dispatches to Application.Current.Dispatcher for ReBuzz API access.
//     On change of IR Wave (or content of the subscribed IWave), rebuilds
//     ConvolutionState on the UI thread and atomically publishes it via a
//     volatile field write (Core §24).
//   • Audio thread reads _state once per Work(), passes refs into two
//     ConvolutionEngine instances (one per channel), processes input,
//     mixes dry/wet/output. No allocation on audio thread (Core §16).
//
// Sample handling:
//   • Buzz Sample is ±32768 int-style float; we normalise to ±1.0 with
//     SCALE = 1/32768 before convolution and de-normalise on output
//     (PedalComp §1). This keeps internal arithmetic at "audio unit"
//     scale where dB and gain calculations are intuitive.
//
// Bypass behaviour:
//   • _state == null  →  engines fill scratch with zeros; only the dry
//     path contributes. With Dry=0 (default) the output is silent — set
//     Dry to a non-zero value to hear input passthrough when no IR
//     is loaded.

using System;
using System.ComponentModel;
using System.Threading;
using System.Windows;
using Buzz.MachineInterface;
using BuzzGUI.Interfaces;

namespace PedalConVerb;

[MachineDecl(Name = "Pedal ConVerb", ShortName = "ConVerb",
             Author = "Pedalrnr", MaxTracks = 0)]
public class PedalConVerbMachine : IBuzzMachine
{
    // dB strings for the Output trim (49 entries, indices 0..48 → −24..+24 dB).
    private static readonly string[] DbLabels24 = new[]
    {
        "-24 dB","-23 dB","-22 dB","-21 dB","-20 dB","-19 dB","-18 dB",
        "-17 dB","-16 dB","-15 dB","-14 dB","-13 dB","-12 dB","-11 dB",
        "-10 dB","-9 dB","-8 dB","-7 dB","-6 dB","-5 dB","-4 dB",
        "-3 dB","-2 dB","-1 dB","0 dB","+1 dB","+2 dB","+3 dB",
        "+4 dB","+5 dB","+6 dB","+7 dB","+8 dB","+9 dB","+10 dB",
        "+11 dB","+12 dB","+13 dB","+14 dB","+15 dB","+16 dB","+17 dB",
        "+18 dB","+19 dB","+20 dB","+21 dB","+22 dB","+23 dB","+24 dB"
    };

    // ── Parameters ───────────────────────────────────────────────────────

    [ParameterDecl(Name = "IR Wave",
                   IsWaveNumber = true,
                   MinValue = 0, MaxValue = 200, DefValue = 0,
                   Description = "Wave-table slot for the impulse response (0 = none)")]
    public int IRWave { get; set; }

    [ParameterDecl(Name = "Dry",
                   MinValue = 0, MaxValue = 127, DefValue = 0,
                   Description = "Dry signal amount (0..127)")]
    public int Dry { get; set; }

    [ParameterDecl(Name = "Wet",
                   MinValue = 0, MaxValue = 127, DefValue = 127,
                   Description = "Wet (convolved) signal amount (0..127, 127 = unity)")]
    public int Wet { get; set; } = 127;

    [ParameterDecl(Name = "Output",
                   MinValue = 0, MaxValue = 48, DefValue = 24,
                   Description = "Output trim (offset; 24 = 0 dB, range ±24 dB)",
                   ValueDescriptions = new[]
                   {
                       "-24 dB","-23 dB","-22 dB","-21 dB","-20 dB","-19 dB","-18 dB",
                       "-17 dB","-16 dB","-15 dB","-14 dB","-13 dB","-12 dB","-11 dB",
                       "-10 dB","-9 dB","-8 dB","-7 dB","-6 dB","-5 dB","-4 dB",
                       "-3 dB","-2 dB","-1 dB","0 dB","+1 dB","+2 dB","+3 dB",
                       "+4 dB","+5 dB","+6 dB","+7 dB","+8 dB","+9 dB","+10 dB",
                       "+11 dB","+12 dB","+13 dB","+14 dB","+15 dB","+16 dB","+17 dB",
                       "+18 dB","+19 dB","+20 dB","+21 dB","+22 dB","+23 dB","+24 dB"
                   })]
    public int Output { get; set; } = 24;

    // ── Host + state ────────────────────────────────────────────────────

    private readonly IBuzzMachineHost _host;

    // The volatile reference is the publish/subscribe channel between UI
    // thread (writer) and audio thread (reader). After publish the state's
    // IR fields are immutable; only the FDL fields are mutated, and only
    // by the audio thread that owns this machine instance (Core §24).
    private volatile ConvolutionState _state;

    // Two engines — one per channel. Persistent stream state lives here;
    // the IR/FDL refs come from _state on each Work().
    private readonly ConvolutionEngine _engineL = new();
    private readonly ConvolutionEngine _engineR = new();

    // Audio-thread scratch (resized lazily; only on UI thread, never inside Work).
    private float[] _scratchInL  = new float[256];
    private float[] _scratchInR  = new float[256];
    private float[] _scratchOutL = new float[256];
    private float[] _scratchOutR = new float[256];

    // Param-change tracking on the UI thread.
    private int    _lastIRWave = -1;
    private IWave  _subscribedWave;
    private PropertyChangedEventHandler _waveHandler;
    private Timer  _pollTimer;

    // ── Construction ─────────────────────────────────────────────────────

    public PedalConVerbMachine(IBuzzMachineHost host)
    {
        _host = host;
        // ParameterGroups is null in ctor (Core §15); start the timer
        // through the dispatcher so first checks run after init completes.
        try { Application.Current?.Dispatcher?.BeginInvoke((Action)StartPollTimer); }
        catch { /* during shutdown / no app context */ }
    }

    // Required by IBuzzMachine; dead code for managed effects (Core §1).
    public void Tick() { }

    private void StartPollTimer()
    {
        _pollTimer = new Timer(_ =>
        {
            try { Application.Current?.Dispatcher?.BeginInvoke((Action)CheckIrChange); }
            catch { }
        }, null, 100, 200);
    }

    // ── IR change detection / rebuild (UI thread) ────────────────────────

    private void CheckIrChange()
    {
        if (IRWave != _lastIRWave)
        {
            _lastIRWave = IRWave;
            RebuildIr();
        }
    }

    private void RebuildIr()
    {
        var buzz = _host?.Machine?.Graph?.Buzz;

        // Drop subscription on the previous wave, if any.
        if (_subscribedWave != null && _waveHandler != null)
        {
            try { _subscribedWave.PropertyChanged -= _waveHandler; } catch { }
        }
        _subscribedWave = null;

        if (IRWave <= 0)
        {
            _state = null;
            try { buzz?.DCWriteLine("[Pedal ConVerb] IR Wave = 0 → bypass (passthrough)"); } catch { }
            return;
        }

        IWave wave = null;
        int wavesCount = -1;
        try
        {
            var waves = buzz?.Song?.Wavetable?.Waves;
            wavesCount = waves?.Count ?? -1;
            int idx = IRWave - 1;          // wave slots are 1-based in the parameter, 0-based in the array
            if (waves != null && idx >= 0 && idx < waves.Count) wave = waves[idx];
        }
        catch (Exception ex)
        {
            try { buzz?.DCWriteLine($"[Pedal ConVerb] wave lookup error: {ex.Message}"); } catch { }
        }

        if (wave == null)
        {
            _state = null;
            try { buzz?.DCWriteLine($"[Pedal ConVerb] IR Wave = {IRWave}: slot empty/null (Waves.Count = {wavesCount}) → bypass"); } catch { }
            return;
        }

        int layerCount = -1;
        try { layerCount = wave.Layers?.Count ?? -1; } catch { }
        if (layerCount <= 0)
        {
            _state = null;
            try { buzz?.DCWriteLine($"[Pedal ConVerb] IR Wave = {IRWave}: wave found but has no layers → bypass"); } catch { }
            return;
        }

        // Re-fire on any wave property change (data reload, layer reshape, etc.).
        // Marshalled to dispatcher so the rebuild happens cleanly on the UI thread
        // even if the event arrives elsewhere.
        _subscribedWave = wave;
        _waveHandler = (_, __) =>
        {
            try { Application.Current?.Dispatcher?.BeginInvoke((Action)RebuildIr); }
            catch { }
        };
        try { wave.PropertyChanged += _waveHandler; } catch { }

        // gainLin = 1.0: the IR is taken at its native level; output trim
        // is applied later in the mix path (Output param).
        var newState = IrPrep.Build(wave, gainLin: 1f, dcLog: buzz);

        if (newState == null)
        {
            try { buzz?.DCWriteLine($"[Pedal ConVerb] IR Wave = {IRWave}: build returned null (zero-length layer?) → bypass"); } catch { }
        }
        else
        {
            try { buzz?.DCWriteLine($"[Pedal ConVerb] IR Wave = {IRWave}: built {newState.IrL.NumPartitions} partitions" + (ReferenceEquals(newState.IrL, newState.IrR) ? " (mono)" : " (stereo)")); } catch { }
        }

        // Atomic publish. The audio thread will pick this up on its next
        // Work() — or the one after, depending on timing. Either is fine.
        _state = newState;
    }

    // ── Audio-thread Work() ──────────────────────────────────────────────

    public bool Work(Sample[] output, Sample[] input, int n, WorkModes mode)
    {
        if (n <= 0) return true;

        // Snapshot the published state once per buffer (volatile read).
        var state = _state;

        // Bypass = clean passthrough. When there's no IR loaded, the wet
        // path can't contribute anything; rather than silence the signal
        // (which is what would happen with default Dry = 0), we pass the
        // input through unchanged so the effect "disappears" from the
        // chain. Dry/Wet/Output have no effect in this mode.
        if (state == null)
        {
            if (input != null)
                Array.Copy(input, 0, output, 0, n);
            else
                Array.Clear(output, 0, n);
            return true;
        }

        // Configure engines for this buffer. SetState short-circuits if
        // refs haven't changed, so this is essentially free in steady state.
        _engineL.SetState(state.IrL, state.FdlLReal, state.FdlLImag);
        _engineR.SetState(state.IrR, state.FdlRReal, state.FdlRImag);

        // Resize scratch if the host upped the buffer size. Allocation on
        // the audio thread is technically against Core §16, but n almost
        // never grows past the initial 256 in practice; this is a one-time
        // adjustment, not a per-buffer allocation.
        if (_scratchInL.Length < n)
        {
            _scratchInL  = new float[n];
            _scratchInR  = new float[n];
            _scratchOutL = new float[n];
            _scratchOutR = new float[n];
        }

        // Mix coefficients (recomputed per buffer; ramping not needed for
        // typical user-rate parameter changes).
        const float Scale  = 1f / 32768f;     // PedalComp §1
        float dryLin = Dry / 127f;
        float wetLin = Wet / 127f;            // 127 → unity
        float outDb  = Output - 24;           // unfold the offset
        float outLin = MathF.Pow(10f, outDb / 20f);

        // De-interleave input into the engine's normalised float planes.
        if (input != null)
        {
            for (int i = 0; i < n; i++)
            {
                _scratchInL[i] = input[i].L * Scale;
                _scratchInR[i] = input[i].R * Scale;
            }
        }
        else
        {
            Array.Clear(_scratchInL, 0, n);
            Array.Clear(_scratchInR, 0, n);
        }

        // Convolve.
        _engineL.Process(_scratchInL, 0, _scratchOutL, 0, n);
        _engineR.Process(_scratchInR, 0, _scratchOutR, 0, n);

        // Mix and re-interleave.
        for (int i = 0; i < n; i++)
        {
            float outL = (_scratchInL[i] * dryLin + _scratchOutL[i] * wetLin) * outLin;
            float outR = (_scratchInR[i] * dryLin + _scratchOutR[i] * wetLin) * outLin;
            output[i]  = new Sample(outL / Scale, outR / Scale);
        }

        return true;
    }

    // ── Status-bar value formatting (Pedal Tracker §7.4 pattern) ────────

    public string DescribeValue(IParameter p, int value)
    {
        if (p == null) return null;
        switch (p.Name)
        {
            case "Dry":
            case "Wet":
                // Map 0..127 to a friendly percentage.
                return $"{value * 100 / 127}%";
            case "Output":
                // Match the ValueDescriptions but synthesised so it stays
                // in sync if those strings ever change.
                int db = value - 24;
                return db == 0 ? "0 dB" : (db > 0 ? $"+{db} dB" : $"{db} dB");
            case "IR Wave":
                return value == 0 ? "(none)" : null;   // let host show slot label
            default:
                return null;
        }
    }
}
