# Experiment provenance

Each new formal or empirical output must include:

- Git commit, tag, branch, and dirty-tree state;
- protocol version and exploratory override status;
- configuration, dataset, split, checkpoint, plant, and property hashes;
- Python, OS, CPU, GPU, driver, PyTorch, CUDA, NumPy, and Z3 versions;
- training/evaluation seeds and solver configuration;
- start/end timestamps and output schema version.

Changes that affect interpretation require a new protocol version. Existing
artifacts are preserved and never silently replaced after a bug fix.
