"""Build the standalone integration snapshot and member-4 entry notebooks.

The single notebook embeds every Member-4 runtime module and clean_v2 data.
It does not pretend missing team work is complete. --strict-final requires
the remaining team notebooks to contain actual implementations first.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ['00_eda', '10_emb_sbert', '12_emb_tokens', '20_m1_svm', '22_m3_xgboost',
            '24_m5_textcnn', '25_m6_bilstm', '30_ensemble', '32_imbalance']


def readiness():
    rows = []
    for stem in EXPECTED:
        path = next((ROOT/'notebooks').rglob(stem+'.ipynb'))
        nb = nbf.read(path, 4)
        lines = sum(len(c.source.splitlines()) for c in nb.cells if c.cell_type == 'code')
        rows.append(dict(stage=stem, implemented=lines > 20, code_lines=lines,
                         path=str(path.relative_to(ROOT)).replace('\\', '/')))
    return rows


class EnglishSource(ast.NodeTransformer):
    """Keep executable source explicit while removing legacy non-English prose."""
    def _body(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node
    visit_Module = _body
    visit_FunctionDef = _body
    visit_ClassDef = _body
    def visit_Constant(self, node):
        if isinstance(node.value, str) and any(ord(c) > 127 for c in node.value):
            node.value = 'Input validation failed; inspect data ids, versions, shapes and partitions'
        return node


def build(strict_final=False):
    status = readiness()
    target = ROOT/'Member4_file'/'integration_status.json'
    target.write_text(json.dumps(dict(full_team_submission_ready=all(x['implemented'] for x in status),
                                      member4_ready=True, dependencies=status), indent=2), encoding='utf-8')
    missing = [x['stage'] for x in status if not x['implemented']]
    if strict_final and missing:
        raise RuntimeError('Full team submission blocked by unimplemented stages: '+', '.join(missing))
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}, 'member4_author': 'Cao Taihe 24101734d'}
    nb.cells = [md('''# DataVinci reproducible Member 4 integration snapshot

**Cao Taihe · 24101734d · 2 November 2026**

This single notebook contains all source code and the immutable cleaned data needed to reproduce Member 4's work from a fresh kernel. Run every cell in order. It generates the official grouped splits, trains independent per-fold FastText embeddings, compares three MLP configurations against a TF-IDF logistic-regression reference, saves OOF and held-out predictions, and records environment and timing.

**Submission status:** the supplied team ZIP still contains skeletons for EDA, SVM, XGBoost, multilingual SBERT, token extraction, TextCNN, BiLSTM, ensemble and imbalance analysis. These results are not invented here. This runnable snapshot is not the completed six-model team submission. Rebuild the final integration after those owners deliver executable contributions. The original cleaning audit is provided alongside this file; this notebook uses its frozen clean_v2 output rather than rerunning that separate NLP pipeline.

Use Python 3.13 with `requirements-member4-tested.txt` for the tested environment, or the team's Python 3.11 environment with its original requirements (that combination has not been validated here). The notebook does not install packages or download models. TF-IDF is used only in the baseline. FastText is a separate embedding model trained without holdout text. No test scores are calculated.

The setup writes only into `member4_notebook_run` beneath the working directory. Cached embeddings are reused only when data, splits, parameters and gensim version match.'''),
    code('''import os
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
import sys, types, json, gzip, base64, hashlib
from pathlib import Path
ROOT = Path.cwd() / 'member4_notebook_run'
ROOT.mkdir(parents=True, exist_ok=True)
pkg = types.ModuleType('src')
pkg.__path__ = [str(ROOT / 'src')]
sys.modules['src'] = pkg
print('Output directory:', ROOT)
''')]
    raw = (ROOT/'data/clean_v2/clean_v2.csv').read_bytes()
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    nb.cells += [md('## Frozen input data\nThe embedded file is the exact clean_v2 CSV supplied by the team. SHA-256 is checked before use. Stable tweet ids and group ids are retained.'),
                 code(f"data = gzip.decompress(base64.b64decode({payload!r}))\nassert hashlib.sha256(data).hexdigest() == {hashlib.sha256(raw).hexdigest()!r}\np = ROOT / 'data/clean_v2/clean_v2.csv'\np.parent.mkdir(parents=True, exist_ok=True)\np.write_bytes(data)\nprint('Verified clean_v2 SHA-256:', hashlib.sha256(data).hexdigest())")]
    modules = ['config', 'data', 'cache', 'evaluate', 'fasttext_emb', 'splitting', 'training', 'member4']
    descriptions = {
        'config': 'Paths, label order and data versions', 'data': 'Data loading and schema checks',
        'cache': 'Id-aligned embeddings and versioned predictions', 'evaluate': 'Common evaluation metrics',
        'fasttext_emb': 'Independent scratch-trained subword embeddings',
        'splitting': 'Group-disjoint holdout and five development folds',
        'training': 'Reusable training loop with masks and early stopping',
        'member4': 'MLP tuning, reference comparison and evidence outputs'}
    for name in modules:
        source = (ROOT/'src'/f'{name}.py').read_text(encoding='utf-8')
        tree = EnglishSource().visit(ast.parse(source))
        source = ast.unparse(ast.fix_missing_locations(tree))
        # A module namespace prevents accidental collisions during integration.
        cell = f"# Embedded src/{name}.py\nmodule = types.ModuleType('src.{name}')\nmodule.__file__ = str(ROOT / 'src' / '{name}.py')\nsys.modules['src.{name}'] = module\nexec(compile({source!r}, module.__file__, 'exec'), module.__dict__)\n"
        # Keep source visible over many lines rather than one long escaped line.
        cell = cell.replace(repr(source), 'r\'\'\'\n'+source+"\n'''", 1)
        nb.cells += [md('## '+descriptions[name]), code(cell)]
    nb.cells += [md('''## Run the complete Member 4 experiment

The first outer SGKF split fixes the holdout before modelling. For each MLP candidate and outer fold, a separate grouped inner split selects the epoch count, then the classifier is refitted on the complete outer training partition. Standardization is fitted only on the corresponding training rows. All candidate scores are reported, including unsuccessful refinements. Selection on the same outer CV results introduces tuning optimism; only the future untouched holdout can resolve final generalization.'''),
                 code("summary = sys.modules['src.member4'].run()\nprint('Selected development Macro-F1:', summary['selected_cv']['macro_f1'])\nassert summary['holdout_evaluated'] is False"),
                 md('## Inspect results and reproduce the split audit'),
                 code("import pandas as pd\nfrom IPython.display import display, Image\nfrom src.data import load_clean, load_splits\nfrom src.splitting import validate_splits, split_summary\nclean = load_clean()\nsplits = load_splits(clean=clean)\nvalidate_splits(splits, clean)\ndisplay(split_summary(clean, splits))\ndisplay(pd.read_csv(ROOT / 'Member4_file/evidence/candidate_metrics.csv')[['model','accuracy','macro_precision','macro_recall','macro_f1','train_seconds']])\nfor name in ['m4_comparison.png','m4_oof_confusion.png','m4_learning_curves.png']:\n    display(Image(filename=str(ROOT / 'figures' / name)))"),
                 md('## Team integration acceptance gate\nA successful Member 4 run does not certify missing team stages. The snapshot below records what was available in the supplied archive. Refresh with `Member4_file/build_notebooks.py --strict-final` after receiving complete contributions.'),
                 code('status = '+repr(status)+"\ndisplay(pd.DataFrame(status))\nprint('Full team submission ready:', all(x['implemented'] for x in status))"),
                 md('''## Interpretation and handoff

Use the fixed `folds_v1.csv` for every model. Probability columns are always Neutral, Positive, Negative. `selected_prediction_run` identifies the OOF and holdout probability file for stacking. Its holdout probabilities are available without evaluating holdout labels. Keep weights, embeddings and prediction caches out of the Git upload.

The tested FastText MLP underperformed the sparse linear reference on development Macro-F1. Mean pooling removes token order, and 1,222 development rows provide limited supervision for 185,347 classifier parameters. These are plausible explanations, not causal findings. Class weighting and stronger regularization did not improve the selected score in the planned three-configuration search. Compare frozen multilingual SBERT when Member 5 supplies that cache. Do not claim a gain or a deployment throughput based on this small corpus.

References: scikit-learn StratifiedGroupKFold documentation; PyTorch reproducibility and AdamW documentation; Bojanowski et al. (2017), Enriching Word Vectors with Subword Information. Team sources: project brief, report template, README and clean_v2 cleaning log supplied in the archive.''')]
    nbf.write(nb, ROOT/'DSAI4205_DataVinci_Final.ipynb')
    nbf.write(nb, ROOT/'notebooks/05_final/99_final.ipynb')
    setup = "import sys\nfrom pathlib import Path\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/config.py').exists())\nsys.path.insert(0, str(ROOT))"
    split = nbf.v4.new_notebook(cells=[md('# Official grouped data partitions\nCao Taihe 24101734d. Seed 42. Preserve near-duplicate groups and reserve the holdout before training.'),code(setup),code('from src.data import load_clean\nfrom src.splitting import write_splits, split_summary\nclean = load_clean()\nsplits = write_splits(clean)\ndisplay(split_summary(clean, splits))')], metadata=nb.metadata)
    nbf.write(split, ROOT/'notebooks/01_data/02_splits.ipynb')
    mlp = nbf.v4.new_notebook(cells=[md('# FastText and MLP\nCao Taihe 24101734d. Separate embedding and classifier. All results are development OOF scores. No test evaluation.'), code('import os\nos.environ.setdefault("MKL_THREADING_LAYER", "GNU")\n'+setup),code('from src.member4 import run\nsummary = run()\nsummary')], metadata=nb.metadata)
    nbf.write(mlp, ROOT/'notebooks/03_models/23_m4_mlp.ipynb')
    print('Built portable single notebook and Member-4 entry notebooks; missing team stages:', ', '.join(missing))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strict-final', action='store_true')
    build(p.parse_args().strict_final)
