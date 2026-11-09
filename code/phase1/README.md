# Phase 1 — Legal language model pretraining

Train four language models from scratch on the official legal corpus. Each model generates a 128-word continuation from a 512-word test prompt. The submission archive contains `prediction.json` at its root, keyed by the official prompt IDs.

## Models and saved runs

| Model | Notebook | Architecture | Saved training progress |
| --- | --- | --- | --- |
| 1 | [Train and generate](model1_decoder/train_and_generate.ipynb) | Causal decoder Transformer, 16 layers, full attention, 2,048-token context | The decoder backup used by later phases records 443,086,746 tokens and an unfinished 810M-token schedule. |
| 2 | [Train and generate](model2_encoder_decoder/train_and_generate.ipynb) | Bidirectional encoder with 6 layers, causal decoder with 8 layers and cross-attention, shared token embeddings | The completed batch-4 run processed 499,999,202 total input and target tokens. Its decoder saw 99,128,004 target tokens, including EOS. |
| 3 | [Train to 500M](model3_local_global/train_500m.ipynb) and [continue to 810M](model3_local_global/continue_to_810m.ipynb) | Causal Transformer with a 256-token local attention window and a full-attention layer every fourth layer | The initial 500M predicted-token run was extended by 310M predicted tokens. |
| 4 | [Train and generate](model4_gated_cnn/train_and_generate.ipynb) | Gated causal convolutional language model with 16 layers, kernel size 3, and a dilation cycle of 10 | The completed run processed 500M predicted tokens. |

Each model directory has the `run_config.json` and `progress.json` from its saved run. Progress describes the run at its saved stopping point. The inference checkpoint `best.pt` may come from an earlier validation step. For Models 2–4, the saved `generation_dev.json` and `generation_test.json` identify the checkpoint and decoding settings of those local prediction runs. The accompanying dev scores come from the course evaluator.

The preparation notebook streams the four legal sources, fits a shared 32,768-entry byte-level BPE tokenizer on training excerpts, and writes memory-mapped token streams plus a document index. The causal models train on shifted next-token targets. Model 2 instead builds within-document 512-word source and 128-word continuation pairs. Model 3 uses a 2,048-token context, 16 layers, width 704, and 11 heads. Its 256-token local window is replaced by full causal attention in layers 4, 8, 12, and 16.

Saved outputs appear beneath the matching notebook code cells. The `*_archived_metrics.jsonl` files beside each model notebook contain the full saved metrics. New training runs show progress every 100 steps and at each validation, while `metrics.jsonl` retains the finer-grained records.

## Codabench results

Model 3 at 810M tokens was the highest-scoring finished Phase 1 submission. It ranked 66th with BERTScore F1 of **0.8317**, ROUGE-L F1 of **0.1971**, and BLEU-4 of **0.0551**.

| Submission ID | BERTScore F1 | Run identified from the records |
| --- | ---: | --- |
| 962723 | **0.8317** | Model 3, 810M |
| 958268 | 0.8306 | Unassigned |
| 958208 | 0.8307 | Model 3, 500M |
| 957509 | 0.8105 | Model 2, encoder–decoder |
| 955092 | 0.8142 | Unassigned |
| 954857 | 0.8204 | Unassigned |
| 954558 | 0.8217 | Model 1, greedy decoding |

The screenshots show seven distinct finished submissions. Submission 959789 displays 0.8290 with status Failed and is excluded from the finished count.

## Kaggle setup

1. Use a Kaggle session with Internet and an authorized `HF_TOKEN` secret to access the course dataset `Exploration-Lab/CS-781-Capstone`. Use a GPU for training and generation.
2. For a fresh corpus build, run [the preparation notebook](shared/prepare_corpus.ipynb). It trains the shared 32,768-entry byte-level BPE and saves 810M training tokens, 2M validation tokens, the document index, and held-out prompts. To reuse the existing corpus, extract `phase1_scratch_data/` from the local `prep phase.zip` backup and attach it to Kaggle. Keep `documents.jsonl` with the tokenizer, manifest, token streams, and held-out JSON files. The [prepared-data manifest](shared/prepared_data_manifest.json) describes that backup.
3. Attach [the course evaluator](evaluate_phase1.py) when using the encoder–decoder, local/global Transformer, or gated CNN notebooks. Set the input paths in the notebook if it finds multiple attached copies. For Model 1, set `DATA_DIR` for training. For generation from saved weights, run the setup and model definitions, then set `GENERATION_RUN` in the generation section. The optional recovery helper resumes incomplete temperature-0.1 predictions.
4. For Model 2, prepare the 512-word input and 128-word target pairs or attach the saved `encdec_pairs_512_128` folder. The default batch-4 run uses `encoder_decoder_500m_batch4_trial`; the earlier batch-1 recipe is documented in the training section. The recorded test policy is in [its generation metadata](model2_encoder_decoder/generation_test.json).
5. For Model 3's continuation, attach the completed `model3_local_global_500m` run and the original prepared corpus. The extension starts from the parent's `latest.pt` and skips the tokens consumed by the parent. The later phases use `best.pt` from `model3_local_global_810m_extension`.

## Run order

1. **Prepare the shared corpus.** Run sections 1–4 of the preparation notebook. For local data, set `LOCAL_DATA` to the folder containing `train_supreme_court.json`, `train_high_courts.json`, `train_acts.json`, and `train_tribunals.json`. Keep the entire `PREPARED_OUTPUT` folder, including `dev_prompts.json` and `dev_truth.json`, for reuse.
2. **Train the chosen model.** Run its setup, data, and architecture cells before the training cells. Point Model 1's `DATA_DIR` or the other notebooks' `INPUT_DATA_DIR` to the prepared corpus. For Model 2, run **Tokenizer and paired data** first, or set `INPUT_PAIR_DIR` to the completed pair dataset. The short benchmark and full-training cells are separate in Models 2–4. For a resumed run, set `RESUME_FROM` to the saved run folder where the notebook provides that setting.
3. **Extend Model 3.** Finish the 500M run first. Attach exactly one `model3_local_global_500m` folder with `latest.pt`, `run_config.json`, and `tokenizer.json`, along with the same prepared corpus. Run the continuation setup, data, and correctness checks, then **Continue or resume training**. Save the extension folder separately.
4. **Generate development and test continuations.** Run **Generate and evaluate** for the selected model. The notebook uses its matching development prompts and references. For official test generation, attach `Phase_1_Pretraining/Test/test.json` or let the notebook fetch it. Models 2–4 pin the test data revision in `TEST_REVISION`. Keep separate generation directories for different decoding settings so cached predictions stay tied to their checkpoint and settings.
5. **Export.** Use the archive from **Test submission**. Preserve the checkpoint, tokenizer, and run configuration for later phases, and retain `latest.pt` if you plan to resume.

## Checkpoints and evaluation

[Checkpoint manifest](checkpoint_manifest.json) records the SHA-256 hashes and archive members for the four Phase 1 inference checkpoints reused in later phases. They are in the existing `phase5_kaggle_assets.zip` backup. Use each original training backup when resuming with `latest.pt`. The Model 3 parent resume checkpoint is also saved in the existing `model 3 continuation req` folder. The [shared tokenizer](shared/tokenizer.json) matches all four inference checkpoints. The weights and prepared corpus are stored separately from this code.

Run the evaluator with predictions and references from the same held-out split.

```bash
python evaluate_phase1.py --truth /path/to/dev_truth.json --prediction /path/to/prediction.json
```

The evaluator reports BERTScore F1, ROUGE-L F1, and BLEU-4.
