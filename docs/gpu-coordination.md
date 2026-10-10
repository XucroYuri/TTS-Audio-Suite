# Optional native-priority GPU coordination

The optional coordinator integration protects supported unified TTS text and SRT execution before model construction. It is disabled by default and does not change native TTS source or ComfyUI core.

Set `TTS_AUDIO_SUITE_GPU_COORDINATION_CONFIG` to a private local JSON file. The file contains `enabled`, `resource_group`, and `groups[resource_group]` with `coordinator_url`, a private `token`, and optional `heartbeat_seconds`. The coordinator URL must use literal loopback HTTP. Do not commit this file, tokens, machine paths or private audio.

The coordinator and TTSMore backend must use the same resource group. The production coordinator requires a persistent ownership journal. A missing/invalid capability, unknown or stale native state, heartbeat failure, unsupported engine mode, retained runtime lease, failed worker-tree cleanup or nonzero process CUDA allocation prevents clean acknowledgement.

Supported coordinated execution holds one actual lease around engine construction, reference preprocessing and inference. A heartbeat detects native priority; external workers check cooperative interruption during polling. Execution releases registered target runtimes and confirms GPU allocation cleanup before returning the lease. User cancellation remains distinct from resource preemption; TTSMore resumes only uncommitted lines.

The current deployment scope is verified external checkout execution. In-process modes without a safe cancellation contract are rejected when coordination is enabled; the disabled path keeps existing behavior. CPU fixtures do not establish a production interruption deadline or validate real model migration.

This gate covers supported Suite TTS entry points. Other GPU plugins or mixed workflows remain outside its scope. Use a dedicated protected TTS instance until broader coverage is implemented. Existing native processes require an explicitly scheduled maintenance restart into the external wrapper; this development stage did not perform that deployment.

## TO DO

- [Measure and enforce engine preemption/cleanup deadlines (#26)](https://github.com/XucroYuri/TTS-Audio-Suite/issues/26): validate real external workers, long text and SRT, and add safe cooperative in-process cancellation before admitting those modes.
- [Enforce direct and mixed workflow coverage (#27)](https://github.com/XucroYuri/TTS-Audio-Suite/issues/27): verify multiple callers/instances and reject unprotected allocation paths.
- [Native maintenance validation (TTSMore #44)](https://github.com/XucroYuri/TTS_more/issues/44): verify stable native PID/ports, actual GPU memory reduction and real audio after restoration.
- [Persistent cleanup-fence recovery (TTSMore #45)](https://github.com/XucroYuri/TTS_more/issues/45): independently prove old holders have cleaned up before an operator recovery. Restart, expired tokens and deleting journal files are not cleanup evidence.
