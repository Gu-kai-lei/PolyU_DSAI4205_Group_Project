"""Assemble delivered team notebooks; reject skeletons before creating a file.

This tool is intentionally separate from the validated Member-4 snapshot.
After assembly the output must be executed with Restart and Run All and fixed
for any integration incompatibilities before it can be called final.
"""
from pathlib import Path
import sys, gzip, base64, hashlib
import nbformat
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import readiness

ORDER = ['00_eda','01_cleaning','02_splits','03_baseline','10_emb_sbert','11_emb_fasttext',
         '12_emb_tokens','20_m1_svm','21_m2_knn','22_m3_xgboost','23_m4_mlp',
         '24_m5_textcnn','25_m6_bilstm','30_ensemble','31_label_noise_errors','32_imbalance']


def assemble():
    missing = [x['stage'] for x in readiness() if not x['implemented']]
    if missing:
        raise RuntimeError('Cannot assemble full team submission; missing implementations: '+', '.join(missing))
    portable = nbformat.read(ROOT/'DSAI4205_DataVinci_Final.ipynb', 4)
    # Reuse self-contained module definitions, but not the Member-4 experiment call.
    stop = next(i for i,c in enumerate(portable.cells) if c.cell_type == 'markdown' and c.source.startswith('## Run the complete'))
    cells = portable.cells[:stop]
    cells[0] = nbformat.v4.new_markdown_cell('# DataVinci team integration candidate\nAssembled by Cao Taihe 24101734d. This candidate must pass full execution before submission. Individual contributions below retain their attribution. Install the full team environment, including NLP resources, before running.')
    # Other members may need the raw and audit CSVs. Keep the input data portable.
    for rel in ['data/raw/stock_market_crash_2022.csv','data/clean_v2/clean_v2_all.csv']:
        raw = (ROOT/rel).read_bytes()
        payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
        cells.append(nbformat.v4.new_code_cell(f"payload = gzip.decompress(base64.b64decode({payload!r}))\nassert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(raw).hexdigest()!r}\np = ROOT / {rel!r}\np.parent.mkdir(parents=True, exist_ok=True)\np.write_bytes(payload)"))
    # Also inline additional public source modules delivered by the team.
    embedded = {'config','data','cache','evaluate','fasttext_emb','splitting','training','member4','__init__'}
    for p in sorted((ROOT/'src').glob('*.py')):
        if p.stem not in embedded:
            source=p.read_text(encoding='utf-8')
            cells.append(nbformat.v4.new_code_cell(f"module = types.ModuleType('src.{p.stem}')\nmodule.__file__ = str(ROOT / 'src' / {p.name!r})\nsys.modules['src.{p.stem}'] = module\nexec(compile({source!r}, module.__file__, 'exec'), module.__dict__)"))
    for stem in ORDER:
        path=next((ROOT/'notebooks').rglob(stem+'.ipynb'))
        nb=nbformat.read(path, 4)
        cells.append(nbformat.v4.new_markdown_cell('## Team contribution '+stem))
        for cell in nb.cells:
            if cell.cell_type == 'code' and ('from pathlib import Path' in cell.source and 'sys.path.insert' in cell.source):
                # Replace repeated repository-discovery setup with portable common imports.
                cells.append(nbformat.v4.new_code_cell('from src.config import *\nfrom src.data import *\nfrom src.cache import *\nfrom src.evaluate import *'))
                continue
            copied=nbformat.from_dict(dict(cell))
            if copied.cell_type == 'code':
                copied.outputs=[]; copied.execution_count=None
            cells.append(copied)
    output=ROOT/'DSAI4205_DataVinci_Team_Candidate.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=cells,metadata=portable.metadata), output)
    print('Candidate assembled, not yet accepted:',output)


if __name__ == '__main__':
    assemble()
