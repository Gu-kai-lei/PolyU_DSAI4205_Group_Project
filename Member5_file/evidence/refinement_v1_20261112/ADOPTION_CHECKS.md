# Adopted C3 checks

- Selected default evidence is the original249-fit experiment: mean pipelineF1 .6166861211377567 vs matchedC0 .6069744799094366; later235-fit local experiment remains archived separately.
- Final choice C3/64filters/.0003lr/.5dropout/.01weightdecay/9epochs/seed42/689155params reconciles with historical protocol and final selection. Model class source digest equals the original trained class.
- Original checkpoint SHA256417cc498513a9b174b2cc8a76b03a19f9c0129b4938dafb3d74ebb1a7e604a27; original teamNPZ SHA25638760eb284bc097b702e578d855ca4654da3b2c2e5abb4002083ce889c3bed0d. Both retained locally under selected manifest paths, not in Git.
- Fresh CPU inference on all303 original reserved prediction rows: maxabs3.5762786865234375e-7, identical classes; labels not scored and no formal training performed.
- Current scoped checks58passed1CUDA-onlyskip, including actualC3 inference, wrong checkpoint/archive rejection, evidence-profile mismatch rejection and fresh standard Jupyter kernels without caches/weights.
- No Google/Drive SDK imports or provider paths introduced. Original metadata/audit is historical, not an assertion that newly packaged source was formally retrained.
- Fresh review caught an unrequested replay wrapper with weak legacy budget/retry rules; it was removed before publication. Existing model adoption does not retrain. Original cached inputs are explicitly required; generated features belong to a new experiment.
- Adoption is a working-version choice, not a claim that the practical +.010 threshold or statistical-advantage gate passed. Both original flags remain false.
