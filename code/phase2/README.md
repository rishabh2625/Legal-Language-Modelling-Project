# Phase 2 — Legal named entity recognition

Extract legal entity spans from court documents using the 12 official labels. The output maps each document ID to a list of `[start, end, label]` triples. Offsets are zero-based character offsets with an exclusive end. The course evaluator uses exact span and label matching for strict macro-F1.

The selected Model 1 run uses overlapping 1,024-token windows. It maps characters to BPE states using cumulative UTF-8 byte offsets, adds character and position embeddings, and uses a convolutional head to produce 25 BIO tag scores. The notebook rejoins the window emissions in document order and applies one constrained Viterbi decode per document. The student comparison uses Model 4's first eight convolution blocks with the teacher's fixed CRF transitions.

## Experiments

| Run | Notebook | Local strict macro-F1 | Codabench strict macro-F1 | Submission ID |
| --- | --- | ---: | ---: | --- |
| Model 1 full fine-tuning | [Model 1 comparison](model1_decoder/ner_adaptation_and_crf.ipynb) | 0.4613 | 0.4719 | 974343 |
| Model 1 bottleneck adapters | [Model 1 comparison](model1_decoder/ner_adaptation_and_crf.ipynb) | 0.4589 | 0.4819 | 974398 |
| Model 1 LoRMA+ | [Model 1 comparison](model1_decoder/ner_adaptation_and_crf.ipynb) | 0.4409 | 0.4427 | 974456 |
| Model 1 adapters + CRF | [Model 1 comparison and CRF extension](model1_decoder/ner_adaptation_and_crf.ipynb) | 0.5493 | 0.5281 | 975129 |
| Model 2 encoder adapters | [Model 2 adapters](model2_encoder_decoder/ner_adapters.ipynb) | 0.2701 | 0.2600 | 972975 |
| Model 3 full fine-tuning, V3 | [Model 3 full fine-tuning](model3_local_global/ner_full_finetune_v3.ipynb) | 0.4543 | 0.4682 | 971638 |
| Model 4 distilled student | [Model 4 distillation](model4_gated_cnn/ner_distillation.ipynb) | 0.4293 | 0.4106 | 975772 |
| Model 4 no-KD-loss control | [Model 4 distillation](model4_gated_cnn/ner_distillation.ipynb), with `kd_weight=0.0` | 0.4125 | 0.4311 | 975774 |

Local scores are from the saved validation metrics or the executed Model 1 comparison notebook. Codabench scores are from the Phase 2 screenshot and recorded upload order. The two Model 4 rows are assigned in training order, with distillation before the no-KD-loss control. The screenshot shows a second successful submission at 0.4819 under ID 974442.

## Run on Kaggle

1. Import the notebook for the experiment and enable a GPU. Attach the matching Phase 1 checkpoint, its `tokenizer.json` and `run_config.json`, plus [the course evaluator](evaluate_phase2.py). The [Phase 1 checkpoint manifest](../phase1/checkpoint_manifest.json) identifies the source runs. Each notebook has explicit path overrides for attached files.
2. Attach the official `Phase_2_LNER` data, or enable Internet and provide an authorized `HF_TOKEN` secret so the notebook can download it. Use dataset revision `89edba6a44d6bb9593fd13b6895b5cd3b06eb82e` to reproduce the recorded inputs. The saved configs record file hashes, label order, and the 60-document training and 10-document validation split.
3. For **Model 1**, the notebook contains the full, bottleneck-adapter, and LoRMA+ runs. Run the adapter section before the CRF extension, or restore its saved run files. The extension reads the adapter configuration and validation artifacts, uses a character BIO CRF, and saves to a separate run directory. The archived CRF run stopped after epoch 35 and used its best validation checkpoint for test predictions.
4. For **Model 2**, use the Phase 1 `encoder_decoder_500m_batch4_trial` checkpoint. The NER model reads the encoder, trains bottleneck adapters and a bidirectional character head, and uses overlapping windows for long documents. For **Model 3**, use the Phase 1 810M extension checkpoint with the V3 masked bidirectional convolution head and full backbone fine-tuning.
5. For **Model 4**, extract the local `model4_phase2_inputs.zip` backup and attach its `teacher_crf/` and `model4_phase1/` folders. Run the data-root resolution cell before loading the corpus. The student initializes from Model 4's Phase 1 embeddings, first eight convolution blocks, and final normalization. It trains on gold character labels and cached teacher posteriors. The no-KD-loss control uses a separate `ROOT_OUT` and the same recipe with `CONFIG['kd_weight'] = 0.0`. Both use the teacher's fixed transition scores for decoding.

## Run order by notebook

- **Model 1 comparison and CRF.** Set `BASE_RUN` to the Phase 1 decoder folder if automatic discovery is ambiguous. Run **A1** through **Shared training and evaluation**. The three comparison methods have separate run folders. For the CRF, run **Bottleneck adapters** through validation first, or restore its `config.json` and `best_validation_metrics.json` to the folder used by `CRF_BASELINE`. Then run the CRF setup, capacity check, and the epoch 10, 20, and 35 training cells in order. Use **Validation comparison and test submission** for the selected CRF checkpoint.
- **Model 2 adapters and Model 3 full fine-tuning.** Set `BASE_RUN` and, if resuming, `RESUME_DIR` in **A1**. Run **A1–E1**, followed by **E2** and **E3** to train, then **F1–F3** for held-out scoring and the test archive.
- **Model 4 student.** Point `TEACHER_DIR` to the CRF export containing `best.pt`, `config.json`, `split.json`, `tokenizer.json`, and `evaluate_phase2.py`. Point `MODEL4_DIR` to the Phase 1 gated CNN assets. Run sections 1–6 to validate the teacher split, cache training probabilities, and check capacity. Sections 7–9 train, score, and export the student.

The official data root must contain `Train/fold1-text.json`, `Train/fold1-gold.json`, `Train/fold2-text.json`, `Train/fold2-gold.json`, `Train/labels.txt`, and `Test/fold3-text.json`. Keep the original text so the output character offsets remain valid.

## Checkpoints and evaluation

The [checkpoint manifest](checkpoint_manifest.json) identifies the original backup ZIPs, the selected weights, and the test generation artifacts. Model 4's teacher hash matches the Model 1 adapter + CRF checkpoint. Each run directory contains its saved configuration and validation metrics where those were available in the backups.

Run the evaluator on matching held-out references and predictions.

```bash
python evaluate_phase2.py --truth /path/to/validation_truth.json --prediction /path/to/prediction.json --labels /path/to/Train/labels.txt
```

For Codabench, package `prediction.json` at the root of `submission.zip`.
