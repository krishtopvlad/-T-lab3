using System;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace DC.Compute;

public sealed class HybridPowSolver : IPowSolver
{
    private sealed class RunState
    {
        public readonly PowSolution[] Solutions = new PowSolution[PowSolver.TaskCount];
        public int NextTaskId = -1;
        public int Completed;
        public long TotalAttempts;
    }

    public async Task<PowResult> SolveAsync(
        string seed,
        int difficultyBits,
        int workerCount,
        IProgress<PowProgress>? progress,
        CancellationToken cancellationToken,
        PowTelemetry? telemetry = null)
    {
        ValidateArguments(seed, difficultyBits, workerCount);
        cancellationToken.ThrowIfCancellationRequested();

        var state = new RunState();
        var elapsed = Stopwatch.StartNew();
        telemetry?.Start();

        try
        {
            // One worker loop per requested worker; each loop claims challenges dynamically.
            Task[] workers = Enumerable.Range(0, workerCount)
                .Select(workerId => Task.Run(() => WorkerLoop(
                    workerId, seed, difficultyBits, state, progress, cancellationToken, telemetry)))
                .ToArray();

            await Task.WhenAll(workers).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            elapsed.Stop();

            if (state.Completed != PowSolver.TaskCount || state.Solutions.Any(solution => solution is null))
                throw new InvalidOperationException("The solver finished without producing all 100 solutions.");

            return new PowResult(Array.AsReadOnly(state.Solutions), elapsed.Elapsed,
                state.Completed, workerCount, Interlocked.Read(ref state.TotalAttempts));
        }
        finally
        {
            elapsed.Stop();
            telemetry?.Finish();
        }
    }

    private static void WorkerLoop(
        int workerId,
        string seed,
        int difficultyBits,
        RunState state,
        IProgress<PowProgress>? progress,
        CancellationToken cancellationToken,
        PowTelemetry? telemetry)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int taskId = Interlocked.Increment(ref state.NextTaskId);
            if (taskId >= PowSolver.TaskCount)
                return;

            int lane = telemetry?.BeginWork(taskId, workerId) ?? -1;
            try
            {
                progress?.Report(new PowProgress(taskId, null,
                    Volatile.Read(ref state.Completed), Interlocked.Read(ref state.TotalAttempts)));

                (ulong nonce, long attempts) = SolveChallenge(
                    seed, taskId, difficultyBits, cancellationToken, telemetry, lane);

                cancellationToken.ThrowIfCancellationRequested();
                var solution = new PowSolution(taskId, nonce);
                state.Solutions[taskId] = solution;
                telemetry?.Solved(taskId);

                long recordedAttempts = Interlocked.Add(ref state.TotalAttempts, attempts);
                int recordedCompleted = Interlocked.Increment(ref state.Completed);
                progress?.Report(new PowProgress(taskId, solution,
                    recordedCompleted, recordedAttempts));
            }
            finally
            {
                telemetry?.EndWork(lane);
            }
        }
    }

    private static (ulong Nonce, long Attempts) SolveChallenge(
        string seed,
        int taskId,
        int difficultyBits,
        CancellationToken cancellationToken,
        PowTelemetry? telemetry,
        int lane)
    {
        byte[] prefix = Encoding.UTF8.GetBytes(
            FormattableString.Invariant($"BSU-POW-1|{seed}|{taskId}|"));

        Span<byte> message = stackalloc byte[prefix.Length + 20];
        prefix.AsSpan().CopyTo(message);
        Span<byte> hash = stackalloc byte[32];

        long attempts = 0;
        long reportedAttempts = 0;

        try
        {
            for (ulong nonce = 0; ; nonce++)
            {
                // Poll cancellation periodically to keep the hot loop inexpensive.
                if ((nonce & 1023UL) == 0)
                    cancellationToken.ThrowIfCancellationRequested();

                if (!Utf8Formatter.TryFormat(nonce, message[prefix.Length..], out int digits))
                    throw new InvalidOperationException("Could not encode the nonce.");

                SHA256.HashData(message[..(prefix.Length + digits)], hash);
                attempts = checked(attempts + 1);

                // Telemetry is updated in batches rather than on every hash.
                if (telemetry is not null && (attempts - reportedAttempts) >= 8192)
                {
                    telemetry.AddAttempts(lane, taskId, attempts - reportedAttempts);
                    reportedAttempts = attempts;
                }

                if (HasLeadingZeroBits(hash, difficultyBits))
                    return (nonce, attempts);

                if (nonce == ulong.MaxValue)
                    throw new InvalidOperationException($"Task {taskId} exhausted all unsigned 64-bit nonces.");
            }
        }
        finally
        {
            // Flush the final partial batch, including on cancellation.
            telemetry?.AddAttempts(lane, taskId, attempts - reportedAttempts);
        }
    }

    private static bool HasLeadingZeroBits(ReadOnlySpan<byte> hash, int difficultyBits)
    {
        int completeBytes = difficultyBits / 8;
        for (int index = 0; index < completeBytes; index++)
        {
            if (hash[index] != 0)
                return false;
        }

        int remainingBits = difficultyBits % 8;
        return remainingBits == 0 || (hash[completeBytes] >> (8 - remainingBits)) == 0;
    }

    private static void ValidateArguments(string seed, int difficultyBits, int workerCount)
    {
        if (seed is null || seed.Length != 6 || seed.Any(character => character is < '0' or > '9'))
            throw new ArgumentException("The seed must contain exactly six decimal digits (0–9).", nameof(seed));
        if (difficultyBits is < 1 or > PowSolver.MaximumDifficultyBits)
            throw new ArgumentOutOfRangeException(nameof(difficultyBits), "Difficulty must be between 1 and 24 bits.");
        if (workerCount is < 1 or > PowSolver.MaximumWorkers)
            throw new ArgumentOutOfRangeException(nameof(workerCount), "Worker count must be between 1 and 100.");
    }
}
