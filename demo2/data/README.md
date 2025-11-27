## Dataset Directory Structure

The project never ships live malware. Populate the directories below on your own machine or in an isolated VM:

| Directory | Purpose | Notes |
|-----------|---------|-------|
| `data/raw/` | Original binaries before processing | Keep hashes and metadata here for reproducibility. |
| `data/benign/` | Clean binaries (optionally packed) | Include both plain and obfuscated benign files to avoid bias. |
| `data/obfuscated/` | Malicious binaries with obfuscation | Organize by malware family: `data/obfuscated/<family>/<sample>` |
| `data/processed/` | Serialized feature matrices and metadata produced by `generate_dataset.py` | Contains `.pkl` splits and `feature_metadata.json`. |
| `data/upload/` | Temporary folder for dashboard uploads | Cleared automatically after each prediction. |

### Filling the folders

1. Run `python scripts/create_sample_structure.py` to scaffold the folders and `.gitkeep` files (optional if you already see them).
2. Use `scripts/download_samples.py` to fetch tagged samples from public feeds (MalwareBazaar, VirusShare, Hybrid Analysis).
3. Place benign executables (including DRM/packed binaries) inside `data/benign/`, grouped by application or vendor for easier tracking.
4. Use `scripts/obfuscate_samples.py` to apply additional packing/obfuscation layers when testing new techniques.

> Always work on disposable VMs or sandboxes. Never run the collected binaries on your host OS. Use the guidance in `docs/VM_SETUP_GUIDE.md` for a hardened lab environment.

