# Phase 4 — Court judgment prediction

Predict a binary outcome for each legal case. The notebook writes one `{"id": "...", "pred": 0}` or `{"id": "...", "pred": 1}` record per official test case. The primary metric is macro-F1 over the two outcome classes.

For a causal backbone, the classifier concatenates the mean and final valid hidden states of each window. It projects the window features, adds a learned position projection, and pools valid windows with attention before predicting two outcome logits. The hierarchical comparison adds a bidirectional GRU over ordered windows. The selected Model 1 LoRMA run uses only the final 1,024 tokens.

## Code and inputs

- [Judgment prediction](judgment_prediction.ipynb) preserves the executed notebook for the two ending-only baselines, the head-and-tail and hierarchical comparisons, and Model 1 LoRMA. To reproduce the highest mapped Codabench run, use its setup and shared helpers, then the **Model 1 LoRMA** cells. The saved LoRMA config uses the ending-only representation and attention pooling. The later extension kept the same best dev score as the original run.
- [Remaining models](remaining_models_min10.ipynb) contains the `model2_full_min10` recipe and the other planned experiment sections. The saved Model 2 submission is available. Its local dev score and final run config are absent from the saved results.
- Attach the Phase 1 checkpoints and tokenizer from `phase3_all_model_assets_kaggle.zip`, along with the official `Phase_4_CJPE/` files or the saved shared data and token caches. The remaining-model notebook also supports `phase4_remaining_kaggle_assets.zip`. Use Kaggle Internet and an authorized `HF_TOKEN` secret if downloading the official files.
- The [run manifest](run_manifest.json) identifies the existing run and shared backups. The `runs/` folders here contain verified metadata for the five runs with saved checkpoints and validation predictions.

## Run order

1. Point `DATA_ROOT` to the folder containing `Train/train.json`, `Train/dev.json`, and `Test/test.json`, or allow the notebook to download them. Set individual `MODEL_DIRS` entries if several Phase 1 checkpoints match. Use `RESUME_ROOT` in the main notebook to restore a saved working tree once, then reset it to `None` on subsequent reruns.
2. In **Judgment prediction**, run sections 1–10 to resolve inputs, audit repeated cases, prepare token caches, and define the training and scoring helpers. The notebook writes the course evaluator into its assets directory.
3. Run the experiment you need. Section 11 contains the ending-only baselines, section 12 the head-and-tail and hierarchical comparisons, and section 13 **Model 1 LoRMA**. Within the selected experiment, run the capacity check, training, validation, and test export. Model 1 LoRMA writes under `/kaggle/working/phase4_cjpe/runs/model1_lorma/`.
4. For the separate **Remaining models** notebook, run sections 1–10 before sections 11–13. For Model 4 distillation, provide a compatible trained Phase 4 teacher folder through `TEACHER_DIR` if automatic discovery is ambiguous. The student workflow validates the teacher on the current dev data before caching training logits. Its runs and archives are written under `/kaggle/working/phase4_remaining`.
5. Submit the ZIP printed by the selected run's test export. Keep the run metadata, `best.pt`, and `latest.pt` if you may continue training.

## Results

| Run | Local dev macro-F1 | Codabench macro-F1 | Submission ID |
| --- | ---: | ---: | --- |
| Model 1 full, ending only | 0.7040 | 0.6986 | 999911 |
| Model 3 full, ending only | **0.7091** | 0.7017 | 1000012 |
| Model 3 full, head and tail | 0.6792 | 0.6743 | 1000155 |
| Model 3 full, hierarchical BiGRU | 0.7052 | 0.7076 | 1000896 |
| Model 1 LoRMA, ending only | 0.7076 | **0.7120** | 1001380 |
| Model 2 full, remaining-model notebook | — | 0.5193 | 1003762 |

Codabench mappings use the screenshot, the saved submission timestamps, and notebook export order. The five available notebook exports match the corresponding saved predictions by hash. The screenshot also shows 999910 cancelled and 1000875 still preparing.

The archived overlap audit found matching dev IDs and normalized text for all 1,517 official test cases. The runs use official dev for checkpoint selection, so these dev and Codabench figures reflect overlapping cases.

## Local evaluation

```bash
python evaluate_phase4.py --truth /path/to/dev_truth.json --prediction /path/to/prediction.json
```

The professor's evaluator reports macro-F1 and positive-class precision and recall. The Codabench ZIP contains `prediction.json` at the root.
