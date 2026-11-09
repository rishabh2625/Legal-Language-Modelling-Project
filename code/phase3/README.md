# Phase 3 — Rhetorical role prediction

Assign one of 13 rhetorical role labels to each sentence in a legal judgment. The official domains are Corporate Law (CL) and Information Technology (IT). The notebook preserves document IDs and sentence order and writes one ordered list of labels per document. The primary metric is sentence-level macro-F1 over the combined CL and IT splits.

The notebook trains a shared classifier across both domains using a 512-token input. For the causal backbones, it concatenates the mean hidden state over the target sentence with the final valid state and predicts one role independently for each sentence. Model 2 uses a BOS-seeded decoder query in place of the final-state readout. The distillation experiment trains on the gold roles and the teacher's cached logits.

## Notebook and inputs

- [Rhetorical role experiments](rhetorical_roles_experiments.ipynb) keeps the original cell outputs for training, development evaluation, and test export. It contains full fine-tuning, LoRA, LoRMA, and IA³ for each Phase 1 backbone, followed by distillation.
- Extract the existing `phase3_all_model_assets_kaggle.zip` and attach `phase3_model_assets/` as a Kaggle input. The [run manifest](run_manifest.json) identifies the four checkpoints and their hashes. These are the same Phase 1 checkpoints listed in the [Phase 1 manifest](../phase1/checkpoint_manifest.json).
- Attach [the course evaluator](evaluate_phase3.py). Attach the official `Phase_3_RR/` data or enable Internet and provide an authorized `HF_TOKEN` Kaggle secret. Set `DATA_REVISION` to `89edba6a44d6bb9593fd13b6895b5cd3b06eb82e` for the recorded run. Use `MODEL_DIRS`, `DATA_ROOT`, and `EVALUATOR_PATH` if automatic discovery finds several copies.
- The input window includes up to two previous sentences and one following sentence. The classification head combines the current-sentence mean with a sequence readout. The official training split has 80 documents and 16,845 sentences. The dev split has 10 documents and 2,142 sentences.
- To reproduce the highest mapped Codabench run, use **Model 1 → Full fine-tuning**, including its validation and export cells. For distillation, the recorded teacher is that Model 1 full run. The student initializes from Model 3’s Phase 1 checkpoint and trains with gold labels and the teacher’s cached training logits.

## Run order

1. Set `DATA_ROOT` to the directory containing `Train/CL_train.json`, `Train/IT_train.json`, `Train/CL_dev.json`, `Train/IT_dev.json`, `Train/label_vocab.json`, `Test/CL_test.json`, and `Test/IT_test.json`, or use the download path above. The shared setup resolves all four Phase 1 models, even when you plan to train one.
2. Run **Required files and environment** through **Training and evaluation helpers**. This loads the official sentence lists and prepares the classifier, capacity check, and resume helpers.
3. In the chosen model and method section, run training, validation, and export in order. Direct experiments write under `/kaggle/working/phase3_rr_v2/direct/<model>/<method>/`. To resume in a fresh session, restore the compatible run folder to the same location under `OUTPUT_ROOT` before the training cell. The notebook's export cells require a completed run.
4. For distillation, complete the teacher run before **Knowledge distillation**. Keep the teacher's `best.pt` and run metadata under the same `OUTPUT_ROOT`, then select and cache the teacher, train the Model 3 student, and run its validation and export cells. The recorded teacher is `model1/full`.

## Results

All 17 local scores below match the notebook's checkpoint reload and course evaluator outputs. Fourteen saved test ZIPs were matched to their notebook exports using the SHA-256 of `prediction.json`. The submitter confirmed the Codabench mappings after comparing screenshot rows with download times, upload order, and local scores. A dash indicates that there is no confirmed finished score to list.

| Model and method | Local dev macro-F1 | Codabench macro-F1 | Submission ID |
| --- | ---: | ---: | --- |
| Model 1 LoRA | 0.3776 | 0.4263 | 991134 |
| Model 1 LoRMA | 0.3842 | 0.4148 | 991146 |
| Model 1 IA³ | 0.3689 | 0.3995 | 991204 |
| Model 1 full fine-tuning | 0.4025 | **0.4293** | 991163 |
| Model 2 LoRA | 0.1280 | 0.1338 | 991212 |
| Model 2 LoRMA | 0.1341 | — | — |
| Model 2 IA³ | 0.0950 | — | — |
| Model 2 full fine-tuning | 0.1968 | — | — |
| Model 3 LoRA | 0.3841 | 0.4086 | 991218 |
| Model 3 LoRMA | 0.3887 | 0.4095 | 991222 |
| Model 3 IA³ | 0.3757 | 0.3976 | 991190 |
| Model 3 full fine-tuning | 0.3906 | 0.4246 | 991322 |
| Model 4 LoRA | 0.3415 | 0.3935 | 992206 |
| Model 4 LoRMA | 0.3554 | — | 992216 (failed) |
| Model 4 IA³ | 0.3143 | 0.3478 | 992355 |
| Model 4 full fine-tuning | 0.3570 | 0.3943 | 992218 |
| Model 3 distilled from Model 1 full | **0.4169** | 0.4290 | 992365 |

The later uploads were closer together and included retries. Submission 992217 is marked failed and displays 0.3478, the same value as finished row 992355. The screenshot shows 992362 as finished with a blank score, so it remains unassigned. The [run manifest](run_manifest.json) retains the matching basis, local domain scores, and artifact hashes.

## Evaluation and submission

Use the two held-out reference files and the official label vocabulary.

```bash
python evaluate_phase3.py --truth-cl /path/to/CL_dev.json --truth-it /path/to/IT_dev.json --prediction /path/to/prediction.json --labels /path/to/label_vocab.json
```

For Codabench, use the experiment's generated ZIP with `prediction.json` at the root and save it as `submission.zip`. Keep prediction ZIPs and model weights in the local backups.
