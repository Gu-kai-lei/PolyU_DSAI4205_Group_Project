# First release verification

- Curated from the completed development work onto latest team main 0498603; only Member5 files, its result table and release documentation change.
- Standard Jupyter entry points: Member5_TextCNN.ipynb and Member5_MVP.ipynb. Fresh ipykernel Restart & Run All was executed in temporary repository copies without embedding caches, from both the root and Member5 directory.
- 52 Member5 tests passed; one CUDA-only check skipped on the Apple MPS host. Real CPU/MPS tiny fitting, saved-weight CPU reload and cache/resume tests passed. Notebook schema and code-cell syntax validate.
- All six published OOF files independently reproduce mean Macro-F1 0.605946 (selection) and 0.615469 (C0), class metrics, confusion matrices, conclusions and paired group bootstrap. Original MVP OOF metrics also reconcile.
- No provider SDK imports, account IDs, fixed cloud paths, personal interpreter paths or large model/cache files are included in the release.
- Fresh generated caches use the fixed encoder revision and a local manifest tied to current data/splits; existing unversioned Transformer arrays and existing FastText arrays are rejected. Generated MVP manifests and runs stay below DATA_ROOT/local_artifacts and no longer rewrite published evidence/results.
- Historical full source/weight verification remains explicitly historical. No new formal tuning was run during packaging; publication changes are recorded in PUBLICATION.json.
- Fresh independent review found the historical MVP overwrite and misleading reproduction-link issues; one fixes pass added regression checks and clarified notebook displays.
- Full repository collection still has three existing Member1 import errors (Member1_file/member1/test_baseline.py, Member1_updated1/member1/test_baseline.py, Member1_updated1/member1/test_split_recovery.py). The full repository suite is not claimed green; shared code is unchanged.

Publication: source commit 5718109e03eea17f98da6b5ad0612a34cf80e090 was pushed non-force to main and read back through GitHub. This receipt adds documentation only.
