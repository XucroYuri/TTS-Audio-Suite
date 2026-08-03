# TTS More target-runtime remote branch audit (2026-08-03)

## Scope and method

This audit was performed from fork base
`ce443482cdddd883914f589705196415bb98e332` after `git fetch --all --prune
--tags`.  Every remote-tracking branch visible through `refs/remotes` was
enumerated with its exact object ID.  Ancestry was checked against both the
integration base and `upstream/main`; relevant divergent branches were also
checked with `git cherry -v` and their patches/tests were inspected.

The selection boundary is deliberately narrow: GPT-SoVITS, IndexTTS,
CosyVoice, the TTS More API Bridge, Windows process/runtime behavior,
FFmpeg/toolchain discovery, and isolated-runtime behavior.  New engines and
unproven experiments are not pulled into this integration branch.  No official
ComfyUI or official TTS-engine source is modified.

## Selection result

No new product patch is required from the fetched branches:

- `origin/main` is already an ancestor of the integration base.
- `upstream/main` adds only `2f587b22b32a42a8d1873ac0926136378c9fc44f`
  (side-effect-free nested-module probing).  The fork already has the same
  behavior in `e349dc00fc83c1c70a06366b8e3c9dfb6a6de359`: the implementation is
  `install.py`'s `PathFinder.find_spec` walk and the binding regression test is
  `tests/unit/test_installer_runtime_repairs.py::test_module_available_finds_nested_module_without_importing_parent`.
  Replaying the upstream release commit would also discard the fork's
  `tts_more_targets` installer profile, so it is classified
  **upstream-covered**, not cherry-picked.
- The FFmpeg branch patches are patch-equivalent to fork commits
  `63b2b56440532700b1b9217171918349e11c8baa` and
  `5dd7d0d29bc16cf0857b7f43f740a8e62872f5e3`; behavior tests were added by
  `4537b42aa6eddeb20ac01b0cb89d177a8a97a96b`.
- Bridge/API and isolated-runtime branches are ancestors of the integration
  base.  GPT-SoVITS, IndexTTS, and CosyVoice external-checkout behavior is
  already implemented by the fork's resource registry, API Bridge, and
  isolated subprocess work.

## Remote branch disposition

`origin` is the fork and `upstream` is the official TTS-Audio-Suite repository.
The `origin` and `upstream` rows that share a SHA are still listed separately
because both refs were execution-visible.

| Remote ref | SHA | Disposition | Evidence and reason |
|---|---|---|---|
| `origin/HEAD` | `1d9e0f6c31309c9ad476da3d735b3aa91f61028f` | integrated | Symbolic alias of `origin/main`; the target is an ancestor of the base. |
| `origin/main` | `1d9e0f6c31309c9ad476da3d735b3aa91f61028f` | integrated | Ancestor of the base; fork Windows late-child stabilization is present. |
| `origin/codex/unified-voice-design-and-saving` | `1d52a512069dd086e0c5f32f06421dcf5ec937ab` | upstream-covered | Ancestor of both base and upstream main; unrelated to the three target runtimes. |
| `origin/cosyvoice-continue` | `33643e4f7faebd12fa1505c837d210f9a198c7e3` | upstream-covered | Ancestor of base/upstream; later CosyVoice external-checkout behavior supersedes it. |
| `origin/debug/startup-timing-273` | `354e5ad523b7deb86a9a2db788e260c16b543c34` | deferred | One divergent diagnostics commit; no target-engine or Windows-runner behavior test justifies integration. |
| `origin/dev-xu/comfyui-api-bridge` | `701a5351e0b2b345e8b3a12561764c5443537ef7` | integrated | Ancestor of the base; Bridge resource and route tests cover the public contract. |
| `origin/echo-tts-integration` | `faf263d24295d6948f30d3e301d715baa32c19f8` | upstream-covered | Echo is already in upstream/base history; it is outside the three-engine delivery scope. |
| `origin/feat/omnivoice-builder` | `14fcba3bed1b709f92f82ff65be03c427c00e8a5` | upstream-covered | OmniVoice is already in upstream/base history; no new target-runtime behavior is selected. |
| `origin/feature/combined-granite-echo-testing` | `99ea0bc6f9b45278f94ca10d139720d98849c693` | upstream-covered | Granite/Echo branch is an ancestor; unrelated engine work is not replayed. |
| `origin/feature/dots-tts` | `3a817a25268b8fabf2bd8a0d859a3ae999e554e2` | upstream-covered | Dots is already in upstream/base history; outside target scope. |
| `origin/feature/higgs-audio-transformers5-investigation` | `49baed441ce2d5aef210ef3f32ea2e671f12535e` | rejected | Divergent experimental Higgs/Transformers 5 investigation; not a target engine. |
| `origin/feature/indextts2-engine-implementation` | `306be0ebb75856bc26feacd5ffeeb7ac9dcc42b8` | upstream-covered | `git cherry` marks its remaining patch equivalent; current IndexTTS implementation and Bridge tests supersede it. |
| `origin/feature/isolated-engine-runtimes` | `667d12c37b0b1c4e0c6d089c3c34553b07370d91` | integrated | Ancestor of base/upstream; later fork subprocess isolation builds on it. |
| `origin/feature/seed-per-iteration-experimental` | `53421b28ac14b07bc8a4ff36dc74a36e6557d7dd` | rejected | Divergent experimental seed semantics; unrelated to the runner/Bridge contract. |
| `origin/fix-step-audio-editx-import-error-6139423650818070639` | `f44430b2414d2c9701269623211717f4b4190ced` | rejected | Step Audio EditX-only fix; outside target engines. |
| `origin/fix/ffmpeg-toolchain-and-tts-paths` | `4b8138daad4549451c1f5a06fb5f9a0c849ee200` | upstream-covered | Both branch commits are patch-equivalent to `63b2b56` and `5dd7d0d`; tests are in `4537b42`. |
| `origin/gguf_failed_attempt` | `62eb5cadbb76d6b05dc77d067eb253680e7b9eb6` | rejected | Named failed attempt, divergent, and unrelated to the target runtimes. |
| `origin/gpt-sovits-integration` | `7c0b73aa1401c5e688323f3a369c1cf5d49f15a7` | deferred | See the commit-level table below; useful behavior is covered, while downloader/console changes need separate TDD or conflict with the external-checkout boundary. |
| `origin/jules-capabilities-doc-10075870564657213280` | `4bc6c558dfd2b678b3031a081d83d8dd6d0797b3` | rejected | Agent-capability documentation only; no runtime behavior. |
| `origin/pr-246` | `1c19fbbd752b96dda7ad3e90ce6d603ce24edb2b` | upstream-covered | Ancestor of base/upstream; unrelated voice-cache signal behavior. |
| `origin/resume/higgs-audio-v3-t5` | `03401c0b40921702abd71cf0be339099d4f4cb3e` | upstream-covered | Ancestor of base/upstream but Higgs is outside scope. |
| `origin/temp-new-engine-guides-docs-20260513` | `3286ef815ee51c11f9c40a16b7a514ca9a9e57a6` | rejected | Divergent temporary new-engine documentation; no target behavior. |
| `origin/wip/kugel-transformers5` | `e477233832bee9f8d9ac18a88aa542c01e8285df` | rejected | WIP Kugel/Transformers 5 investigation; outside target scope. |
| `upstream/HEAD` | `2f587b22b32a42a8d1873ac0926136378c9fc44f` | upstream-covered | Symbolic alias of `upstream/main`; its one new behavior is already covered by `e349dc0`. |
| `upstream/main` | `2f587b22b32a42a8d1873ac0926136378c9fc44f` | upstream-covered | Version 5.6.3 behavior is present with a fork regression test; no unsafe release-wide replay. |
| `upstream/codex/add-voxcpm-engine` | `a2513527c94d5a2fae393cddb5ced7019bd172fb` | rejected | Divergent VoxCPM engine/training work; outside target scope. |
| `upstream/codex/audio8-engine` | `73a282406e6a34f6949cbcd5e8c0eaad8e1274e0` | rejected | Divergent Audio8 engine; outside target scope. |
| `upstream/codex/dramabox-chatterbox-v3` | `517d11ed4c4e193155e3ecd39aa6d2627712a8e6` | upstream-covered | Ancestor of base/upstream; unrelated engines. |
| `upstream/codex/tada-engine` | `35cbcaca714faa3d34bfbfd8663ce87bf7cb86a5` | rejected | Divergent TADA engine/tooltips; outside target scope and not on upstream main. |
| `upstream/codex/unified-voice-design-and-saving` | `1d52a512069dd086e0c5f32f06421dcf5ec937ab` | upstream-covered | Same covered SHA as the fork ref. |
| `upstream/cosyvoice-continue` | `33643e4f7faebd12fa1505c837d210f9a198c7e3` | upstream-covered | Same covered CosyVoice SHA as the fork ref. |
| `upstream/debug/startup-timing-273` | `354e5ad523b7deb86a9a2db788e260c16b543c34` | deferred | Same divergent generic diagnostics commit as the fork ref. |
| `upstream/echo-tts-integration` | `faf263d24295d6948f30d3e301d715baa32c19f8` | upstream-covered | Same covered Echo SHA as the fork ref. |
| `upstream/feat/omnivoice-builder` | `14fcba3bed1b709f92f82ff65be03c427c00e8a5` | upstream-covered | Same covered OmniVoice SHA as the fork ref. |
| `upstream/feature/combined-granite-echo-testing` | `99ea0bc6f9b45278f94ca10d139720d98849c693` | upstream-covered | Same covered Granite/Echo SHA as the fork ref. |
| `upstream/feature/dots-tts` | `3a817a25268b8fabf2bd8a0d859a3ae999e554e2` | upstream-covered | Same covered Dots SHA as the fork ref. |
| `upstream/feature/higgs-audio-transformers5-investigation` | `49baed441ce2d5aef210ef3f32ea2e671f12535e` | rejected | Same rejected Higgs experiment as the fork ref. |
| `upstream/feature/indextts2-engine-implementation` | `306be0ebb75856bc26feacd5ffeeb7ac9dcc42b8` | upstream-covered | Same patch-equivalent old IndexTTS ref as the fork. |
| `upstream/feature/isolated-engine-runtimes` | `667d12c37b0b1c4e0c6d089c3c34553b07370d91` | integrated | Same covered isolated-runtime ancestor as the fork. |
| `upstream/feature/seed-per-iteration-experimental` | `53421b28ac14b07bc8a4ff36dc74a36e6557d7dd` | rejected | Same rejected seed experiment as the fork ref. |
| `upstream/fix-step-audio-editx-import-error-6139423650818070639` | `f44430b2414d2c9701269623211717f4b4190ced` | rejected | Same out-of-scope Step Audio EditX fix. |
| `upstream/gguf_failed_attempt` | `62eb5cadbb76d6b05dc77d067eb253680e7b9eb6` | rejected | Same named failed attempt as the fork ref. |
| `upstream/jules-capabilities-doc-10075870564657213280` | `4bc6c558dfd2b678b3031a081d83d8dd6d0797b3` | rejected | Same documentation-only ref as the fork. |
| `upstream/pr-246` | `1c19fbbd752b96dda7ad3e90ce6d603ce24edb2b` | upstream-covered | Same covered ancestor as the fork. |
| `upstream/resume/higgs-audio-v3-t5` | `03401c0b40921702abd71cf0be339099d4f4cb3e` | upstream-covered | Same covered Higgs ancestor as the fork; no new integration. |
| `upstream/temp-new-engine-guides-docs-20260513` | `3286ef815ee51c11f9c40a16b7a514ca9a9e57a6` | rejected | Same temporary documentation branch as the fork. |
| `upstream/wip/kugel-transformers5` | `e477233832bee9f8d9ac18a88aa542c01e8285df` | rejected | Same WIP Kugel experiment as the fork. |

## GPT-SoVITS branch commit decisions

| Commit | Disposition | Reason |
|---|---|---|
| `5eb772159e965fa7c37bd6a9b5eaee05dced6ecc` | rejected | Branch-local implementation plan is superseded by the resource-registry/API-Bridge design. |
| `c598229e94baa2ad2267a94512346ada2c997573` | upstream-covered | GPT-SoVITS nodes/runtime exist in the base; fork commits from `dccdda0` onward use the official runtime contract and add deterministic tests. |
| `8f2a881a5602cd65deea009ec11d114425309580` | upstream-covered | Checkout binding is covered by `59ced47`, `0d5c67f`, and later isolated registered-runtime commits. |
| `094264cfc15656ff84f205a27eeca4d18e16f3bb` | rejected | Plugin-owned downloader/model layout conflicts with the machine-local registry and official external-checkout boundary. |
| `94b6a8810f99ea57091fc8934bea854653378681` | deferred | Broad all-engine console rewrite has no focused GBK behavior test; any real console failure requires a separate RED/GREEN fix. |
| `7c0b73aa1401c5e688323f3a369c1cf5d49f15a7` | upstream-covered | Registered source roots for IndexTTS and CosyVoice are covered by the API Bridge and isolated subprocess implementation/tests. |

## Explicit exclusions

Fix11/Fix12 shared-root deletion, quarantine, rollback, and recursive-removal
experiments were not fetched into this branch, inspected for reuse, or merged.
Live CUDA validation was not moved into hosted CI.  This audit contains no
resource-registry values, model paths, fixture contents, or private commands.
