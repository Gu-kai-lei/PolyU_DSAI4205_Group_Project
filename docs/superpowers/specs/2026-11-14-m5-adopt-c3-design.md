# Adopt the user's intended C3 version

User explicitly corrects the desired release: screenshot identifies textcnn_refinement_colab_latest_20261112, selected-pipeline meanF1 .616686 vs its C0 .606974, final C3/64 filters/lr.0003/dropout.5/wd.01/9epochs/seed42/689155params. Adopt as current working version; don't describe the later .605946 C1 result as selected. Recorded point estimates favour this run, but different protocols and inspected OOF prevent confirmed superiority claims; all C0 delta intervals cross0, practical gate remains false.

Preserve the generic Jupyter and provider-free fixes already on main. Add immutable selected-version manifest, publish the earlier small evidence, redirect default evidence/model loader to the original C3 and record actual historical candidates C0/C1/C3/C7. The later six-candidate local experiment remains archived as a negative result. Don't retrain or reinterpret workflow F1 as final fixed C3 accuracy.

Local adopted weights and team NPZ copied by new names only after hash verification; no shared-file overwrite. Published model loader validates model/data/split/cache hashes and the original final-weight digest, loads CPU by default, preserves probability order. Weights remain ignored per team policy. Default notebook review requires no weights/cache; opt-in saved-model prediction is ordinary Python/Jupyter, no cloud SDK.

Recompute earlier OOF metrics/bootstrap, reload existing C3 on CPU and match all303 recorded probabilities/classes without scoring labels. Historical complete249-fit audit remains historical. No new training experiment; tiny tests only. After fresh review and scoped tests, non-force push corrected main under user's standing authorization.

Review ruling: remove the unrequested legacy replay wrapper rather than advertise unsafe retry/budget guarantees. Adoption does not retrain; future experiments stay separate under the existing durable research workflow. Document obtaining exact audited caches instead of promising regenerated caches are identical.
