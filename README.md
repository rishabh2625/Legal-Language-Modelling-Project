# CS781 Legal NLP Capstone

The notebooks train four compact legal language models from scratch, then adapt their checkpoints for six downstream tasks. Each phase guide identifies the notebooks, the assets they expect, the order to run the cells, and the evaluation and submission files.

## Running the notebooks

1. Import the notebook for the experiment into Kaggle and enable a GPU for training and generation. The recorded runs used two T4 GPUs. Use the same GPU count when resuming a checkpoint whose training state records it.
2. Attach the official phase data with its original folder structure. To let a notebook download the data, enable Internet and add an authorized Hugging Face token as the Kaggle secret named `HF_TOKEN`. Each phase guide lists its file layout and any dataset revision required for the recorded run.
3. Train or restore the Phase 1 checkpoint before starting a downstream experiment. The [Phase 1 checkpoint manifest](code/phase1/checkpoint_manifest.json) records the archived filenames, run folders, and hashes. References to backup ZIPs in the phase guides identify separately stored inputs. The repository contains the notebooks, evaluators, and run metadata.
4. Run setup and shared helper cells before the experiment cells. Kaggle inputs are read-only. Use the notebook's working directory for new outputs, and follow the phase-specific restore procedure when bringing back a saved run.
5. Save the run folder before ending a session. Preserve `latest.pt` and its metadata for resuming, and keep `best.pt` with its matching configuration and tokenizer for inference. Use the test export from the intended experiment for submission. Its ZIP must have `prediction.json` at the root.

To run the scoring scripts outside Kaggle, install their dependencies in a Python environment. Phases 1 and 6 use BERTScore, and Phase 6 also uses ROUGE Score. The classification evaluators that import scikit-learn can use the same environment.

```bash
python -m pip install torch bert-score rouge-score scikit-learn
```

Run the evaluator command from the corresponding phase folder using labelled validation references. The first BERTScore run may need network access to fetch the scoring model.

## Architecture, methods, and submissions

The four Phase 1 models share a custom byte-level BPE tokenizer. Later tasks start from the corresponding Phase 1 checkpoint and add a task-specific head or reuse the language-model output for summarization. The phase guides document the individual runs, their settings and notebook entry points, and the available submission IDs and scores.

| Phase guide | Best recorded submission | Architecture and method |
| --- | --- | --- |
| [1. Legal language model pretraining](code/phase1/README.md) | Model 3 at 810M tokens · 0.8317 BERTScore F1 | Causal Transformer with local and periodic full attention. Continue the 500M checkpoint on the unused 310M targets, then decode a 128-word continuation. |
| [2. Legal named entity recognition](code/phase2/README.md) | Model 1 adapters + CRF · 0.5281 strict macro-F1 | Freeze the decoder, train bottleneck adapters and a character tagger, and use BIO-constrained Viterbi decoding to recover exact character spans. |
| [3. Rhetorical role prediction](code/phase3/README.md) | Model 1 full fine-tuning · 0.4293 macro-F1 | Pack the target sentence with neighbouring sentences and classify its role from target-token and sequence representations. |
| [4. Court judgment prediction](code/phase4/README.md) | Model 1 LoRMA · 0.7120 macro-F1 | Adapt the causal decoder and classify from the ending window. The guide also covers head-and-tail and hierarchical window comparisons. |
| [5. Legal statute identification](code/phase5/README.md) | Model 3 full fine-tuning · 0.3743 macro-F1 | Attend over judgment windows separately for each statute and use a development-selected sigmoid threshold. |
| [6. Abstractive legal summarization](code/phase6/README.md) | Model 1 full fine-tuning · 0.8473 BERTScore F1 | Prompt a causal decoder with head-and-tail context and train only on the summary continuation. |
| [7. Prior case retrieval](code/phase7/README.md) | Model 4 distilled from Model 3 · 0.3810 micro-F1@7 | Train a reduced gated CNN with labelled case pairs and teacher scores, then rank by a blend of full-text TF-IDF and dense similarity. |
