# MOSS Clip loader / NumPy compatibility review

- Review range: `29891d2..978a7786d76d32b5065751376aa6437d54be8042`
- Review status: **READY**
- Reviewed source commits: `151e5661e1bbf21fae3e3d3b678808087369acbe`, `978a7786d76d32b5065751376aa6437d54be8042`

## Scope and findings

The loader change in `nodes/audio/vocal_removal_node.py` adds each removed NumPy
scalar alias only when it is absent from `numpy.__dict__`; existing aliases,
including `np.bool`, retain object identity. The fix does not touch `nodes.py` or
the MOSS Clip implementation, so it does not fake-register `MossClipStagingNode`.
The probe still imports the real node module and verifies the registered ID.
Optional dependency stubs remain limited to the existing probe harness, and the
parent-process isolation assertions remain intact.

The first fix commit's test incorrectly required `np.bool` to be unchanged even
when it was absent (NumPy 1.26.4). Commit `978a778` corrected this to require
the alias to be available and to require identity preservation only when an
alias existed before loading.

## Verification evidence

All commands ran against the exact review range with no source/test edits.

| Environment | Result |
|---|---|
| `J:\ComfyUI-aki-v2\python\python.exe`, Python 3.12.10, NumPy 1.26.4 | focused registration `4 passed`; full unit `335 passed, 2 skipped` |
| `F:\venvs\comfyui-tts\Scripts\python.exe`, Python 3.12.3, NumPy 2.5.1 | full unit run 1 and run 2 each `335 passed, 2 skipped`; `pip check` clean |
| Isolated Python 3.12.10, NumPy 2.2.6 | full unit `335 passed, 2 skipped` |

`py_compile` passed for the changed source and test files, and
`git diff --check 29891d2..978a778` passed.

The J portable environment's `pip check` remains non-zero because of
pre-existing unrelated package state: missing OpenCV packages, `aiortc`/`av`,
`inference-*`/`aiohttp`, and `torchscale`/`timm` requirements. No NumPy-related
conflict was reported there.
