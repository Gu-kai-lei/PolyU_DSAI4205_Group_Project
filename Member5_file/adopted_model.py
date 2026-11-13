"""The explicitly adopted C3 checkpoint; no cloud SDK, download or training."""
from pathlib import Path
import hashlib
import inspect
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Member5_file'))


def selected_version(manifest_path=None):
    value=json.loads(Path(manifest_path or ROOT/'Member5_file/selected_version.json').read_text())
    relative=Path(value['evidence_directory'])
    if relative.is_absolute() or len(relative.parts)!=1 or relative.name in ['.','..']:raise ValueError('Invalid selected evidence directory')
    weight=Path(value['model']['weights_relative_path'])
    if weight.is_absolute() or '..' in weight.parts or weight.parts[0]!='models':raise ValueError('Invalid model relative path')
    evidence=ROOT/'Member5_file/evidence'/relative
    final=json.loads((evidence/'final_selection.json').read_text())
    cfg=value['model']
    if any(cfg[k]!=final[k] for k in ['candidate','epochs','parameters']):raise ValueError('Selected model and evidence final choice differ')
    protocol=json.loads((evidence/'protocol.json').read_text())
    if any(cfg[k]!=v for k,v in protocol['candidates'][cfg['candidate']].items() if k not in ['max_epochs','patience']):raise ValueError('Selected model config differs from trained protocol')
    conclusion=json.loads((evidence/'conclusion.json').read_text())
    for key,field in [('pipeline_mean_macro_f1','selected_mean'),('matched_control_mean_macro_f1','control_mean'),('practical_target_met','practical_target_met'),('statistical_advantage_confirmed','statistical_advantage_confirmed')]:
        if value[key]!=conclusion[field]:raise ValueError('Selected version score/decision differs from its evidence')
    benchmark=json.loads((evidence/'benchmark.json').read_text())
    if value['historical_candidates_executed']!=benchmark['candidate_ids']:raise ValueError('Adopted candidate set differs from recorded execution')
    archive=Path(value['team_predictions']['relative_path'])
    if archive.is_absolute() or '..' in archive.parts or archive.parts[0]!='preds':raise ValueError('Invalid team archive relative path')
    return value


def default_evidence_directory():return ROOT/'Member5_file/evidence'/selected_version()['evidence_directory']


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_adopted_model(weights_path=None):
    from src.config import DATA_ROOT,CLEAN_DIR
    version=selected_version();cfg=version['model']
    path=Path(weights_path) if weights_path is not None else DATA_ROOT/cfg['weights_relative_path']
    if not path.is_file():raise FileNotFoundError('Adopted C3 checkpoint missing. Place the audited local artifact at '+str(path)+'; its checksum is in selected_version.json.')
    if _hash(path)!=cfg['weights_sha256']:raise ValueError('Adopted C3 weight checksum mismatch')
    protocol=json.loads((default_evidence_directory()/'protocol.json').read_text())
    if _hash(CLEAN_DIR/'clean_v2.csv')!=protocol['data_sha256'] or _hash(ROOT/'splits/folds_v1.csv')!=protocol['split_sha256']:raise ValueError('Adopted model task data/split differs')
    import torch
    from mvp import TextCNN
    from safetensors.torch import load_file
    source=inspect.getsource(TextCNN).strip()
    if hashlib.sha256(source.encode()).hexdigest()!=cfg['architecture_sha256']:raise ValueError('Adopted TextCNN architecture differs')
    with torch.random.fork_rng(devices=[]):model=TextCNN(cfg['input_dim'],filters=cfg['filters'],dropout=cfg['dropout'])
    model.load_state_dict(load_file(str(path)))
    if sum(p.numel() for p in model.parameters())!=cfg['parameters']:raise ValueError('Adopted parameter count differs')
    model.eval();model.requires_grad_(False)
    return model


def predict_cached(ids,weights_path=None,device='cpu'):
    from src.config import EMB_DIR
    from src.cache import load_embedding
    from v2_training import probabilities,validate_device
    manifest=json.loads((ROOT/'Member5_file/evidence/embedding_manifest.json').read_text())
    for name,checksum in manifest['files'].items():
        if _hash(EMB_DIR/name)!=checksum:raise ValueError('Adopted model frozen cache checksum differs: '+name)
    device=validate_device(device);model=load_adopted_model(weights_path).to(device)
    x=load_embedding('tokens_multi',ids=ids);mask=load_embedding('tokens_mask',ids=ids)
    return probabilities(model,x,mask,device)


def verify_team_archive(path=None):
    from src.config import DATA_ROOT
    import numpy as np
    version=selected_version();metadata=version['team_predictions']
    path=Path(path) if path is not None else DATA_ROOT/metadata['relative_path']
    if _hash(path)!=metadata['sha256']:raise ValueError('Adopted team archive checksum mismatch')
    with np.load(path,allow_pickle=False) as saved:
        if str(saved['run_id'])!=version['original_run_id'] or int(saved['seed'])!=metadata['seed'] or list(saved['classes'])!=version['model']['classes']:raise ValueError('Adopted team archive identity differs')
        return {name:saved[name].copy() for name in saved.files}
