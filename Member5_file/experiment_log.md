# Member 5 experiment log — 2026-11-07

- Based on main 0498603; clean_v2 and official folds_v1 retained unchanged. Reused member 4 training and team cache/evaluation interfaces.
- Installed an isolated Python 3.12 CPU environment. Exact package versions are recorded in evidence/environment.json and requirements-macos-lock.txt.
- Extracted the frozen multilingual contextual encoder at pinned revision 4328cf26390c98c5e3c738b4460a05b95f4911f5. Development-only length selection yielded 82 tokenizer positions. All 1,525 IDs exported with checksums.
- Compared six sentence vectors against SentenceTransformer.encode with the same revision/truncation; maximum absolute difference 1.01e-7.
- Fixed TextCNN architecture and 12-epoch selection budget before completing the two representation experiments. Grouped inner epoch selection followed by fresh outer refits; seed 42. Test probabilities exported without test scores.
- Initial FastText run completed predictions but aborted when a GUI Matplotlib backend attempted to draw. Reproduced with a minimal plot; selected Agg for headless figures. The complete rerun reproduced the same metrics exactly. Interrupted evidence is retained but excluded from the team results table.
- Completed FastText TextCNN: pooled OOF Macro-F1 0.536124; contextual TextCNN: 0.579576; official baseline: 0.581996. No improvement over baseline is claimed.
- Five focused tests passed. Read-only code review found no blocking outer-fold/test leakage or interface bug. Unsupervised FastText includes inner-stop texts within outer-train; documented as an inner-selection limitation.
- All reported metrics, cache IDs, group boundaries, probabilities and saved-weight inference are independently reconciled by verify_artifacts.py. See evidence/verification.json for the actual check result.
- Local CPU execution measured; live Google Colab execution not yet verified. The team's test set was previously evaluated by another member; this contribution does not restore a pristine holdout.
