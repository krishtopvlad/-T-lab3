using System.Buffers.Text;
using System.Security.Cryptography;
using System.Diagnostics;
using System.Numerics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Buffers.Binary;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;

namespace DC.Compute;

public sealed class CooperativePowSolver : IPowSolver
{
    // Размер одного диапазона nonce, который получает воркер.
    private const int ChunkSize = 10_000;

    private sealed class Puzzle
    {
        public byte[] Prefix = [];

        public long NextChunk = -1;

        public int State;

        public int Started;

        public int Active;
    }

    private sealed class Run
    {
        public readonly Puzzle[] Puzzles = new Puzzle[PowSolver.TaskCount];

        public readonly PowSolution[] Solutions = new PowSolution[PowSolver.TaskCount];

        public int NextTask = -1;

        public int Completed;

        public long TotalAttempts;

        public bool UseAvx2;
    }

    public enum KernelMode { Auto, HashData, Avx2 }

    public static KernelMode Mode { get; set; } = KernelMode.Auto;

    public static string AutoKernelName => UseAvx2Auto ? "Avx2" : "HashData";

    // Подготавливает задачи, запускает воркеры и собирает результаты.
    public async Task<PowResult> SolveAsync(string seed, int difficultyBits, int workerCount,
        IProgress<PowProgress>? progress, CancellationToken cancellationToken, PowTelemetry? telemetry = null)
    {
        if (seed is null || seed.Length != 6 || seed.Any(c => c is < '0' or > '9'))
            throw new ArgumentException("The seed must contain exactly six decimal digits (0–9).", nameof(seed));
        if (difficultyBits is < 1 or > PowSolver.MaximumDifficultyBits)
            throw new ArgumentOutOfRangeException(nameof(difficultyBits), "Difficulty must be between 1 and 24 bits.");
        if (workerCount is < 1 or > PowSolver.MaximumWorkers)
            throw new ArgumentOutOfRangeException(nameof(workerCount), "Worker count must be between 1 and 100.");
        cancellationToken.ThrowIfCancellationRequested();

        var run = new Run
        {
            UseAvx2 = Mode switch
            {
                KernelMode.HashData => false,
                KernelMode.Avx2 => Avx2.IsSupported && Vector256.IsHardwareAccelerated,
                _ => UseAvx2Auto
            }
        };
        for (int id = 0; id < run.Puzzles.Length; id++)
            run.Puzzles[id] = new Puzzle
            {
                Prefix = Encoding.UTF8.GetBytes(FormattableString.Invariant($"BSU-POW-1|{seed}|{id}|"))
            };

        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var token = linked.Token;
        var elapsed = Stopwatch.StartNew();
        telemetry?.Start();
        Task[] workers = [];
        try
        {
            workers = Enumerable.Range(0, workerCount).Select(workerId =>
                Task.Factory.StartNew(() =>
                {
                    try { WorkerLoop(workerId, difficultyBits, run, progress, token, telemetry); }
                    catch { linked.Cancel(); throw; }
                }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();

            try { await Task.WhenAll(workers).ConfigureAwait(false); }
            catch
            {
                if (cancellationToken.IsCancellationRequested)
                    throw new OperationCanceledException(cancellationToken);
                Exception? fault = workers.Select(t => t.Exception?.InnerException)
                    .FirstOrDefault(e => e is not null and not OperationCanceledException);
                if (fault is not null) ExceptionDispatchInfo.Capture(fault).Throw();
                throw;
            }

            cancellationToken.ThrowIfCancellationRequested();
            elapsed.Stop();
            if (run.Completed != PowSolver.TaskCount || run.Solutions.Any(s => s is null))
                throw new InvalidOperationException("The solver finished without producing all 100 solutions.");
            return new PowResult(Array.AsReadOnly(run.Solutions), elapsed.Elapsed, run.Completed,
                workerCount, Interlocked.Read(ref run.TotalAttempts));
        }
        finally
        {
            elapsed.Stop();
            linked.Cancel();
            try { await Task.WhenAll(workers).ConfigureAwait(false); } catch { /* уже обработано выше */ }
            telemetry?.Finish();
        }
    }

    // Сначала воркер берёт отдельные задачи, затем помогает завершать оставшиеся.
    private static void WorkerLoop(int workerId, int bits, Run run,
        IProgress<PowProgress>? progress, CancellationToken ct, PowTelemetry? telemetry)
    {
        // Этап 1: получаем новые задачи из общей очереди.
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            int id = Interlocked.Increment(ref run.NextTask);
            if (id >= PowSolver.TaskCount) break;
            Work(id, workerId, bits, run, progress, ct, telemetry);
        }

        // Этап 2: когда новых задач нет, подключаемся к ещё не решённым.
        var spin = new SpinWait();
        while (Volatile.Read(ref run.Completed) < PowSolver.TaskCount)
        {
            ct.ThrowIfCancellationRequested();
            int best = -1, bestActive = int.MaxValue;
            for (int i = 0; i < PowSolver.TaskCount; i++)
            {
                Puzzle p = run.Puzzles[i];
                if (Volatile.Read(ref p.State) != 0) continue;
                int active = Volatile.Read(ref p.Active);
                if (active < bestActive) { best = i; bestActive = active; }
            }
            if (best < 0) break;
            Work(best, workerId, bits, run, progress, ct, telemetry);
            spin.SpinOnce(-1);
        }
    }

    // Обрабатывает одну задачу: получает чанки и прекращает поиск после нахождения nonce.
    private static void Work(int id, int workerId, int bits, Run run,
        IProgress<PowProgress>? progress, CancellationToken ct, PowTelemetry? telemetry)
    {
        Puzzle puzzle = run.Puzzles[id];
        int lane = telemetry?.BeginWork(id, workerId) ?? -1;
        Interlocked.Increment(ref puzzle.Active);
        try
        {
            if (Interlocked.Exchange(ref puzzle.Started, 1) == 0)
                progress?.Report(new PowProgress(id, null,
                    Volatile.Read(ref run.Completed), Interlocked.Read(ref run.TotalAttempts)));

            while (Volatile.Read(ref puzzle.State) == 0)
            {
                ct.ThrowIfCancellationRequested();
                // Каждый воркер атомарно получает следующий, ещё не проверенный чанк.
                long chunk = Interlocked.Increment(ref puzzle.NextChunk);
                bool found = run.UseAvx2 && chunk >= 1
                    ? SearchChunkAvx2(puzzle.Prefix, chunk, bits, out ulong nonce, out int tried)
                    : SearchChunkHashData(puzzle.Prefix, checked((ulong)chunk * ChunkSize), bits, out nonce, out tried);

                Interlocked.Add(ref run.TotalAttempts, tried);
                telemetry?.AddAttempts(lane, id, tried);

                if (found && Interlocked.CompareExchange(ref puzzle.State, 1, 0) == 0)
                {
                    var solution = new PowSolution(id, nonce);
                    run.Solutions[id] = solution;
                    telemetry?.Solved(id);
                    int completed = Interlocked.Increment(ref run.Completed);
                    progress?.Report(new PowProgress(id, solution, completed,
                        Interlocked.Read(ref run.TotalAttempts)));
                }
            }
        }
        finally
        {
            Interlocked.Decrement(ref puzzle.Active);
            telemetry?.EndWork(lane);
        }
    }


    // Таблица ускоряет подготовку четырёх последних цифр nonce для AVX2.
    private static readonly uint[] DigitTable = BuildDigitTable();

    private static uint[] BuildDigitTable()
    {
        var t = new uint[ChunkSize];
        for (int d = 0; d < t.Length; d++)
            t[d] = (uint)(('0' + d / 1000) << 24 | ('0' + d / 100 % 10) << 16 | ('0' + d / 10 % 10) << 8 | ('0' + d % 10));
        return t;
    }

    private static readonly uint[] K =
    [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ];
    

    // При первом запуске сравнивает скорость двух ядер и запоминает более быстрое.
    private static readonly bool UseAvx2Auto = Avx2.IsSupported && Vector256.IsHardwareAccelerated && AutotuneAvx2();

    private static bool AutotuneAvx2()
    {
        try
        {
            byte[] prefix = Encoding.UTF8.GetBytes("BSU-POW-1|000000|99|");
            const int bits32 = 32;
            double Best(bool avx)
            {
                double best = double.MaxValue;
                for (int rep = 0; rep < 3; rep++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    for (long c = 1; c <= 3; c++)
                    {
                        if (avx) SearchChunkAvx2(prefix, c, bits32, out _, out _);
                        else SearchChunkHashData(prefix, (ulong)c * ChunkSize, bits32, out _, out _);
                    }
                    best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalSeconds);
                }
                return best;
            }
            Best(true); Best(false);
            return Best(true) <= Best(false);
        }
        catch { return false; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    // Стандартное ядро: проверяет один nonce за раз через SHA256.HashData.
    private static bool SearchChunkHashData(byte[] prefix, ulong start, int bits, out ulong nonce, out int tried)
    {
        int prefixLen = prefix.Length;
        Span<byte> msg = stackalloc byte[prefixLen + 20];
        Span<byte> hash = stackalloc byte[32];
        prefix.CopyTo(msg);
        Utf8Formatter.TryFormat(start, msg[prefixLen..], out int digits);
        int len = prefixLen + digits;
        int shift = 32 - bits;
        ulong current = start;
        for (int n = 0; n < ChunkSize; n++)
        {
            SHA256.HashData(msg[..len], hash);
            if ((BinaryPrimitives.ReadUInt32BigEndian(hash) >> shift) == 0)
            {
                nonce = current;
                tried = n + 1;
                return true;
            }
            current++;
            int pos = len - 1;
            while (pos >= prefixLen && msg[pos] == (byte)'9') { msg[pos] = (byte)'0'; pos--; }
            if (pos >= prefixLen) msg[pos]++;
            else
            {
                msg[prefixLen] = (byte)'1';
                msg[len] = (byte)'0';
                len++;
            }
        }
        nonce = 0;
        tried = ChunkSize;
        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<uint> Rotr(Vector256<uint> x, [ConstantExpected] byte n) =>
        Vector256.ShiftRightLogical(x, n) | Vector256.ShiftLeft(x, 32 - n);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    // AVX2-ядро: проверяет восемь nonce одновременно с помощью SIMD.
    private static bool SearchChunkAvx2(byte[] prefix, long chunk, int bits, out ulong nonce, out int tried)
    {
        Span<byte> block = stackalloc byte[64];
        prefix.CopyTo(block);
        Utf8Formatter.TryFormat((ulong)chunk, block[prefix.Length..], out int stemDigits);
        int p = prefix.Length + stemDigits;
        int totalLen = p + 4;
        block[totalLen] = 0x80;

        Span<Vector256<uint>> w = stackalloc Vector256<uint>[64];
        Span<uint> c = stackalloc uint[16];
        for (int i = 0; i < 14; i++)
            c[i] = ((uint)block[4 * i] << 24) | ((uint)block[4 * i + 1] << 16) | ((uint)block[4 * i + 2] << 8) | block[4 * i + 3];
        c[15] = (uint)(totalLen * 8);
        for (int i = 0; i < 16; i++) w[i] = Vector256.Create(c[i]);

        int ia = p >> 2;
        int a = p & 3;
        byte shr = (byte)(8 * a), shl = (byte)(32 - 8 * a);
        uint cA = c[ia], cB = c[ia + 1];
        ref uint table = ref MemoryMarshal.GetArrayDataReference(DigitTable);
        ref uint kRef = ref MemoryMarshal.GetArrayDataReference(K);
        int shift = 32 - bits;
        Vector256<uint> iv0 = Vector256.Create(0x6a09e667u);

        for (int d = 0; d < ChunkSize; d += 8)
        {
            Vector256<uint> packed = Vector256.LoadUnsafe(ref table, (nuint)d);
            w[ia] = Vector256.Create(cA) | Vector256.ShiftRightLogical(packed, shr);
            if (a != 0) w[ia + 1] = Vector256.Create(cB) | Vector256.ShiftLeft(packed, shl);

            // Подготовка расписания SHA-256 для восьми nonce сразу.
            for (int t = 16; t < 64; t++)
            {
                Vector256<uint> x = w[t - 15], y = w[t - 2];
                Vector256<uint> s0 = Rotr(x, 7) ^ Rotr(x, 18) ^ Vector256.ShiftRightLogical(x, 3);
                Vector256<uint> s1 = Rotr(y, 17) ^ Rotr(y, 19) ^ Vector256.ShiftRightLogical(y, 10);
                w[t] = w[t - 16] + s0 + w[t - 7] + s1;
            }

            Vector256<uint> va = Vector256.Create(0x6a09e667u), vb = Vector256.Create(0xbb67ae85u),
                vc = Vector256.Create(0x3c6ef372u), vd = Vector256.Create(0xa54ff53au),
                ve = Vector256.Create(0x510e527fu), vf = Vector256.Create(0x9b05688cu),
                vg = Vector256.Create(0x1f83d9abu), vh = Vector256.Create(0x5be0cd19u);
            // 64 раунда сжатия SHA-256 выполняются векторно.
            for (int t = 0; t < 64; t++)
            {
                Vector256<uint> S1 = Rotr(ve, 6) ^ Rotr(ve, 11) ^ Rotr(ve, 25);
                Vector256<uint> ch = vg ^ (ve & (vf ^ vg));
                Vector256<uint> t1 = vh + S1 + ch + Vector256.Create(Unsafe.Add(ref kRef, t)) + w[t];
                Vector256<uint> S0 = Rotr(va, 2) ^ Rotr(va, 13) ^ Rotr(va, 22);
                Vector256<uint> maj = (va & vb) | (vc & (va | vb));
                vh = vg; vg = vf; vf = ve; ve = vd + t1; vd = vc; vc = vb; vb = va; va = t1 + S0 + maj;
            }

            // Проверяем ведущие нулевые биты по первому 32-битному слову хеша.
            uint mask = Vector256.Equals(Vector256.ShiftRightLogical(va + iv0, shift), Vector256<uint>.Zero)
                .ExtractMostSignificantBits();
            if (mask != 0)
            {
                int dd = d + BitOperations.TrailingZeroCount(mask);
                nonce = (ulong)chunk * ChunkSize + (ulong)dd;
                tried = dd + 1;
                return true;
            }
        }
        nonce = 0;
        tried = ChunkSize;
        return false;
    }
}
