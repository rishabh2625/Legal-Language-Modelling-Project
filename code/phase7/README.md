# Phase 7 — Prior case retrieval

Retrieve seven unique prior case IDs for each query from the official candidate pool for that split. The output maps each query ID to its ordered list of seven IDs. The primary metric is Micro-F1@7.

The dense retriever shares a backbone between queries and candidates. It mean-pools valid token states per window, projects each window to a normalised 256-dimensional vector, and combines the window vectors before computing cosine similarity. Training uses labelled positives and lexical hard negatives. The Model 4 student also learns from Model 3's candidate score distribution.

## Notebooks and Kaggle inputs

| Notebook | Runs | Existing Kaggle asset ZIP |
| --- | --- | --- |
| [Part A](part_a_full_and_lorma.ipynb) | Model 3 full and Model 1 LoRMA | `phase7_part_a_assets.zip` |
| [Part B](part_b_ia3_and_distillation.ipynb) | Model 2 IA³ and Model 4 distillation | `phase7_part_b_complete_assets.zip` |
| [Part C](part_c_lora_and_adalora.ipynb) | Model 3 LoRA and AdaLoRA | `phase7_part_c_assets.zip` |

Use the official `Phase_7_PCR/` data at revision `93b0059be756c6c461f3a225de25aabda7d5f84a`. Attach the official files or enable Internet and an authorized `HF_TOKEN` secret. The notebooks save each run separately and export the test ranking from its selected dev checkpoint and policy.

The final runs filter exact query self-matches, use lexical hard negatives during training, and encode up to three 512-token windows per case. All six exports use the dev-selected `hybrid_tfidf` policy, which combines 75% normalized full-text TF-IDF and 25% normalized dense score. The [dev lexical baselines](dev_lexical_baselines.json) record the standalone lexical scores.

## Run order

1. Supply all six official files under `DATA_DIR`, or let the notebook download them. The pairs are `Train/train.json` and `Train/train_candidates.json`, `Dev/dev.json` and `Dev/dev_candidates.json`, and `Test/test.json` and `Test/test_candidates.json`. Set `DATA_REVISION` explicitly in Parts A and C if downloading the recorded revision. Use the same revision across all parts.
2. Attach the Phase 1 asset ZIP for the notebook or use `MODEL_ASSET_DIRS` to locate extracted folders. The notebooks verify archived Phase 1 asset hashes, including when folders are set explicitly. Use the checkpoint files identified by the [Phase 1 manifest](../phase1/checkpoint_manifest.json) for the recorded runs.
3. In each part, run **Settings**, **Original Phase 1 definitions**, **Data and format checks**, and **Retrieval, official evaluator, and resumable training** before selecting an experiment. These sections build or reuse token caches and prepare the lexical features used in training and ranking.
4. Run the experiment's capacity, training, validation, and test export cells in order. Part A contains the full Model 3 retriever and Model 1 LoRMA, Part B contains Model 2 IA³ and Model 4 distillation, and Part C contains Model 3 LoRA and AdaLoRA. Keep matching run signatures and `latest.pt` to resume under `/kaggle/working/phase7_pcr_final/runs/<run-name>/`.
5. Use the chosen run's `submission.zip`. For offline validation, keep `dev_gold.json`, `dev_predictions.json`, and `dev_candidate_ids.json` from the same run.

### Reproduce the winning teacher–student run

1. In **Part A**, train the Model 3 full retriever and export its teacher checkpoint. The saved teacher export is `phase7_final_part_a_teacher.zip`. The existing combined Part B asset ZIP already contains that checkpoint.
2. In **Part B**, set `TEACHER_RUN = 'model3_full'` if more than one compatible teacher is available, then run **Select and cache the teacher**. A separately attached export contains `phase7_teacher_manifest.json` and `phase7_teacher.pt`. The student retains five Model 4 convolution blocks and learns from the labeled pairs and teacher scores. The [teacher–student comparison](runs/model4_distilled/teacher_student.csv) records both runs.
3. Run the student test export. The [run manifest](run_manifest.json) records the teacher checkpoint hash used by the student and the result archives for both stages.

## Results

| Run | Dev Micro-F1@7 | Codabench Micro-F1@7 | Submission ID |
| --- | ---: | ---: | --- |
| Model 3 full fine-tuning (teacher) | 0.3747 | 0.3768 | 1039492 |
| Model 1 LoRMA | 0.3780 | 0.3780 | 1039490 |
| Model 2 IA³ | 0.3591 | 0.3615 | 1039967 |
| Model 4 distilled from Model 3 full | 0.3780 | **0.3810** | 1039970 |
| Model 3 LoRA | **0.3824** | 0.3792 | 1039543 |
| Model 3 AdaLoRA | 0.3780 | 0.3798 | 1039541 |

The six final dev scores were reproduced with the professor's evaluator. The earlier Model 3 LoRA submission 1036900 scored 0.2209 and ranked each query's own ID first. Its test predictions are preserved in the original archive, and its hash is recorded in the manifest.

## Local evaluation

```bash
python evaluate_phase7.py --truth /path/to/dev_gold.json --prediction /path/to/dev_predictions.json --candidate-ids /path/to/dev_candidate_ids.json
```

For Codabench, use `submission.zip` with `prediction.json` at the root.
