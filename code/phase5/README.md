# Phase 5 — Legal statute identification

Predict a set of statute IDs for each case. The notebook writes one `{"id": "...", "pred": [1, 5]}` record per test case. The primary metric is macro-F1 across the 100 statute labels. The secondary metrics are micro-F1 and subset accuracy.

The main classifier represents a judgment with up to four 1,024-token windows. Each statute has a learned attention query over the window features and a separate output score. The selected Model 3 run uses binary cross-entropy over the 100 logits. At inference, the notebook applies sigmoid and the development-selected threshold to produce the final set of statute IDs.

## Code and inputs

- [Statute identification](statute_identification.ipynb) contains the executed Model 3 full, Model 1 LoRMA, and reduced Model 4 distillation workflows. The main runs use hierarchical chunk representations and label attention. The Model 4 student retains eight convolution blocks and uses the saved Model 3 full checkpoint as its teacher.
- [Model 2 encoder readout](model2_encoder_readout.ipynb) compares the hybrid encoder/decoder-query readout with encoder-only adapters. The saved Codabench prediction ZIP matches the encoder-only export. This run stopped after three epochs and selected the step-zero checkpoint at threshold 0.30. The export predicted all 100 statute IDs for every test case.
- Attach the official `Phase_5_LSI/` files or download them with an authorized `HF_TOKEN` secret. The original `phase5_kaggle_assets.zip` supplies the Phase 1 checkpoints and tokenizer. Model 2 uses the `encoder_decoder_500m_batch4_trial` checkpoint. The [run manifest](run_manifest.json) lists the saved run backups and checkpoint hashes.
- To reproduce the highest mapped Codabench run, use **Model 3 → Full fine-tuning** and its validation and export cells. The saved test export used threshold **0.20**. For distillation, keep both Model 3 and Model 1 run folders available for the teacher-selection cell. The saved student signature identifies Model 3 as the selected teacher.

## Run order

1. Point `DATA_ROOT` to the folder containing `Train/train.json`, `Train/dev.json`, and `Test/test.json`, or let the notebook download them. Use `MODEL_DIRS` to point to the Phase 1 run folders when automatic discovery is ambiguous. The audit and token-cache cells must run before training or restoring a run.
2. In **Statute identification**, run sections 1–9 before the experiment cells. Then use section 10 for Model 3 full fine-tuning or section 11 for Model 1 LoRMA. Run training, validation, analysis, and test export in the selected section. The main notebook writes under `/kaggle/working/phase5_lsi/runs/<run-name>/`. Rerun training with the same configuration and saved `latest.pt` to continue a paused run.
3. For **Model 4 distillation**, make a trained teacher run available to the main notebook before section 12. Set `TEACHER_RUN_NAME = 'model3_full'` to reproduce the recorded teacher choice. Run teacher selection, student training, validation, and export in order.
4. In **Model 2 encoder readout**, run sections 1–8 to resolve the Phase 1 encoder–decoder checkpoint, prepare token caches, and check for compatible saved runs. Run section 9 for the hybrid baseline and section 10 for the encoder-only control. Set `HYBRID_BASELINE_DIR` if you are importing a saved hybrid run. Use the control's test export for the recorded Model 2 submission.
5. Export from the selected `best.pt` using its stored development threshold. The recorded Model 3 export used 0.20. Keep the exported prediction and validation metadata with the run.

## Results

| Run | Local dev macro-F1 | Codabench macro-F1 | Submission ID |
| --- | ---: | ---: | --- |
| Model 3 full fine-tuning | **0.3771** | **0.3743** | 1013521 |
| Model 1 LoRMA | 0.3335 | 0.3302 | 1012963 |
| Model 2 encoder-only adapters | 0.0714 | 0.0713 | 1012980 |
| Model 4 distilled from Model 3 full | 0.3199 | — | 1013531 (failed, displayed 0.3177) |

All four saved prediction ZIPs match notebook exports by hash. The screenshot IDs are inferred from the saved exports, timestamps where available, and the dev scores. Submission 1013050 is shown as submitted with a blank score and remains unassigned. The Model 1 local figure is from the course evaluator on the exported validation predictions.

## Local evaluation

```bash
python evaluate_phase5.py --truth /path/to/dev_truth.json --prediction /path/to/prediction.json
```

Use the notebook's recorded label vocabulary and a threshold chosen on the dev split. The Codabench ZIP contains `prediction.json` at the root.
