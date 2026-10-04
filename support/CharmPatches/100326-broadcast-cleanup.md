# Retain delayed Charm++ broadcasts

This campaign branch records an external-runtime fix. It does not change the GH equations, filtering, horizon finder, or control laws. SpECTRE base: `827c08ed808e74a8c332e641c853b69812adb105`.

Charm++ base: `c92494d89be08e1f4c1d556da34ac74b5bd2eb0f` (installed 8.0.0). The accompanying patch is Charm commit `e254c647154a8f6dafc656e9e1d6075b52dd9313`, branch `codex/broadcast-cleanup-100326`. Build it with the same MPI/SMP production options as the installed runtime, then configure SpECTRE with `CHARM_ROOT` pointing at that separate build.

The old broadcast cleanup advances its retained head past missing epochs. An arriving older epoch produces a negative signed offset that becomes an unsigned size, causing capacity doubling to overflow and loop forever. A live stalled horizon restart had epoch7, head24, offset−17 and overflowed capacity0. The patch ages only the contiguous received prefix, retains the previous cleanup delay, and fails explicitly if an unexpectedly expired epoch still arrives. It neither discards a late message nor changes array layouts.

The exact queue-source regression reproduces the old hang and verifies ordered delivery after a gap, retention of the late message, ordinary cleanup, and 1,000 randomized arrival sequences. It can be built independently from `tests/charm++/broadcastCleanup` in the patched Charm source, then checked with CTest. The original time-based retention assumption for arbitrarily long migrations is not replaced by a proof of globally safe migration garbage collection.

For the campaign, `build_100326i` builds `SystemUtilities` against `charm_100326i`. A directory containing only its `libSystemUtilities.so` is prepended to `LD_LIBRARY_PATH` for new trial launches. This composes the runtime correction with the already-built scientific variants without replacing their GH/horizon libraries or any active file. Dynamic relocation checks verify that all other SpECTRE libraries still resolve from each executable's original build. This requires the unchanged Charm class layout and compiler ABI; actual evolution remains the integration test. The original installed runtime remains read only.

All campaign paths are under `/home/glovelace/Codes/spectre/build_codex_data`. The run/build ledger, native reproducer results, stack samples, and runtime validation are in `campaign_100326` and `build_100326i` under that directory.
