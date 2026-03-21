# deepiri-gpu-utils

## What this is

`deepiri-gpu-utils` is the **Deepiri Hybrid LLM Build & Dev Toolkit**.

This branch contains the **package skeleton + CLI stub**. Later branches will
implement real GPU/device detection and setup logic.

## CLI

Run:

```bash
deepiri-gpu --help
deepiri-gpu detect --json
deepiri-gpu doctor --json
deepiri-gpu setup --device auto --dry-run
deepiri-gpu build-args --json
deepiri-gpu validate --json
deepiri-gpu ollama recommend --json
```

