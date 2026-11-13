# Member 5 first release

User authorizes publishing the previously completed work to main as a first delivery, then improving performance later. Explicit constraint: ordinary Jupyter-compatible code, no Google Drive/Colab imports or cloud authentication dependencies.

Release from latest origin/main (0498603), scope Member5_file and its results only. Preserve other members' data/splits/shared modules. Curate a new commit rather than publish the entire development history with private Drive IDs and machine paths. Existing experimental branches and ignored local weights/caches remain untouched.

Primary notebook: ordinary Python/ipykernel Jupyter, repo-root discovery from root or Member5_file, current interpreter subprocesses, CPU/MPS/CUDA optional selection. Default Restart & Run All reviews and independently recomputes committed evidence, with no training/download/cache prerequisite. Full reproduction is explicit opt-in and uses the unchanged frozen encoder/TextCNN and grouped protocol. No fixed /content, private IDs, personal interpreter paths, or cloud SDK imports.

Deliver original SBERT/FastText TextCNN MVP code/evidence plus latest complete local diagnostic study, clearly reporting selected pipeline .605946 vs same-protocol C0 .615469 and failed performance-advantage gate. Never claim fixed final C1 has an independent test score. Large caches, weights, local environment and full archive stay ignored; report their contract and reproducible generation.

Two checks: execute the generic notebook in a fresh standard Jupyter kernel and recompute all six OOF metrics from committed files; real CPU/MPS trainer checks remain green. Historical full source/weight audit was performed before cloud-path cleanup; label it historical rather than claim the cleaned release source was used in that training. New experiments get new signatures.

User's explicit push-to-main instruction supersedes earlier no-merge/no-push preference. Prepare/test/review the concrete release first; fetch again and use non-force fast-forward push, respecting branch protection if present. Do not bypass protection or rewrite others' history.
