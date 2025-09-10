# LatentFlow
A concept model architecture that I would like to see implemented, tested and expanded.
Please note, I am open sourcing the concept because I don't have the time to pull this research off on my own, any PR's are welcome.
This is the absolute first iteration of the concept so there are still many rough edges.

# PerceiverIO + HRM + Liquid Nets — Research Prototype

**Repository README (research-oriented)**

---

## Title

**PerceiverIO–HRM–LTC Hybrid** — A multimodal, hierarchical, and continuous-time architecture for scalable long-context reasoning and stable temporal control

**Authors / Maintainers:** Talon Bernard van Vuuren
**Contact / Maintainer email:** talbvvtrueuser9174@gmail.com

---

# Abstract

This document describes a research-oriented reference implementation and experiment plan for a hybrid architecture that combines three complementary literatures:

1. **Perceiver / Perceiver IO**: modality-agnostic, scalable encoder/decoder that compresses large and heterogeneous inputs into a fixed-size latent workspace and supports flexible query-based decoding.
2. **Hierarchical Reasoning Model (HRM)**: a multi-timescale planner/worker recurrent design that separates slow, abstract planning from fast, detailed execution.
3. **Liquid Time-Constant (LTC) networks**: continuous-time recurrent cells with learnable time constants that produce stable, adaptive temporal dynamics suitable for streaming, irregularly sampled, or event-driven data.

I provide a conceptual specification, implementation scaffold, training recipes, prioritized experiments, ablations, evaluation metrics, and practical engineering notes intended to let a researcher or contributor reproduce, extend, and test the hybrid concept on reasoning, long-context, multimodal, and continuous-control tasks.

---

# 1. Motivation & Goals

Modern AI problems increasingly require systems that can (a) consume very large heterogeneous inputs (long documents, videos, multimodal sensor streams), (b) perform structured, multi-step reasoning and planning, and (c) operate robustly over irregular or continuous time. Transformers and standard RNNs excel at some of these properties but fail when all are needed at once (e.g., long-range memory + continuous-time dynamics + hierarchical decision-making).

This hybrid aims to combine strengths:

* **Perceiver IO** for compressing and unifying arbitrary inputs and enabling flexible outputs.
* **HRM** for explicit hierarchical, iterative reasoning separating planning (abstract) from execution (detailed).
* **LTC** for numerically stable continuous-time execution dynamics in the low-level worker.

Primary research goals:

* Test whether the hybrid improves sample efficiency on structured reasoning tasks.
* Validate robust long-context performance (language and multimodal).
* Demonstrate improved handling of irregularly sampled / streaming data.
* Provide a modular codebase for ablation studies and further research.

---

# 2. What this repo contains (conceptual, not yet implemented)

* A research README describing architecture, training, and experiments (this file).
* Reference PyTorch scaffold: a minimal working codebase that implements the Perceiver IO encoder/decoder, a two-tier HRM controller (HLP + LLW), and an LTC-based LLW. The scaffold includes toy task scripts (sudoku, graph-search), dataset drivers, and evaluation tools.
* Experiment configurator and example hyperparameter files for quick reproduction of key baselines.
* Templates for adding new tasks and datasets.

---

# 3. Architecture (High-level)

```
[ Input modalities (text, image, audio, video, sensors) ]
                 |
             (embed tokens)
                 |
        Perceiver IO cross-attention
                 |
        Perceiver latent workspace L (N_lat x D_lat)
                 |
       +--------------------------------+
       |                                |
 High-Level Planner (HLP)        Perceiver Latents (shared)
  (slow, recurrent)                     |
       |                                |
  plan tokens P  <----------------->  cross-attend
       |                                |
 Low-Level Worker (LLW)  <---- cross-attention ---->  L
 (continuous-time LTC cell(s))
       |
   outputs / internal writes
       |
  Perceiver IO decoder (output queries)
       |
   structured outputs (tokens, maps, controls)
```

Key flow notes:

* Inputs are flattened and projected into token embeddings for Perceiver cross-attention.
* Perceiver produces a fixed-size latent workspace `L` that the HRM uses as the shared context.
* The HRM comprises: (1) a High-Level Planner (HLP) that operates at a slow timescale and emits plan tokens, and (2) a Low-Level Worker (LLW) implemented as one or more LTC cells that integrate continuously between planner updates.
* Output queries (Perceiver IO decoder) may read both `L` and HRM internal states to produce structured outputs such as next-token predictions or control commands.

---

# 4. Formal component descriptions

## 4.1 Perceiver IO (front-end + decoder)

* Input: sequence of heterogeneous tokens `X = {x_i}` (variable length, modality-labeled).
* Input embedding: per-token linear projection + optional modality-specific positional encodings (time, 2D grid, waveform index).
* Cross-attend to a latent `L ∈ R^{N_lat×D_lat}` via a standard cross-attention block (queries=latent, keys/values=inputs).
* Apply a stack of latent self-attention blocks to `L`.
* Decoder: for each requested output `q_j` (could be autoregressive token queries or structured queries), cross-attend `q_j` to `L`. Decoder may also cross-attend HRM outputs.

## 4.2 Hierarchical Reasoning Module (HRM)

* **High-Level Planner (HLP)**:

  * Input: summary view of `L` (pooled vector(s) or attended queries) + previous HLP state `h_H^{t-1}`.
  * Computation: small recurrent transformer or LSTM producing new planner state `h_H^t` and a set of plan tokens `P^t = {p_k}`.
  * Timescale: slow — updates every `K` inference ticks (K≥1). HLP can also run iteratively for `M` planning steps per input.

* **Low-Level Worker (LLW)**:

  * Implementation: one or several LTC cells parameterized to accept plan tokens `P^t` and context from `L`.
  * Dynamics: continuous-time integration `dh_L/dt = f(h_L, P^t, context(L), t; θ)` where `f` is implemented by the LTC cell.
  * Execution: the LLW runs `n_micro_steps` (or integrates for a time window) for each HLP plan step, attending back to `L` at chosen micro-step intervals.

* **HLP ⇄ LLW interface**:

  * Cross-attention: HLP can attend to LLW states (pooled) and LLW can attend to HLP plan tokens.
  * Gating: learned gates modulate how much HLP plans influence LLW dynamics.

## 4.3 Output layer

* Structured Perceiver IO queries obtain outputs via cross-attention to `L` and optionally HRM pooled states.
* For autoregressive tasks (language modeling), decode iteratively with token queries that include previous token embeddings and HRM context.

---

# 5. Pseudocode (high level)

```python
# high-level forward pass pseudocode
L = perceiver_encoder(inputs)  # cross-attend inputs -> latent
h_H = HLP.init_state()
h_L = LLW.init_state()
for t in range(T_planner_steps):
    # Planner step (slow)
    plan_tokens, h_H = HLP.step(L_summary(L), h_H)

    # Worker integrates continuously
    for _ in range(n_micro_steps):
        h_L = LLW.integrate(h_L, plan_tokens, L)
        # optionally write back to L or produce intermediates

    if iterate:  # optional iterative refinement
        L = perceiver_writeback(L, h_L)

outputs = perceiver_decoder(L, h_H, h_L)
return outputs
```

Notes: `LLW.integrate` is an LTC cell call that does continuous dynamics for a fixed micro-step size (or uses an ODE solver). `perceiver_writeback` is an optional cross-attention from LLW/HRM into the Perceiver latent to support iterative refinement.

---

# 6. Implementation notes & dependencies (recommended stack)

* Primary framework: **PyTorch** (recommended for easier LTC integration and debugging)
* Optional: JAX/Flax for research-scale Perceiver implementations if you prefer
* Suggested Python packages (conda / pip):

  * `torch` (>=1.13)
  * `numpy`, `tqdm`, `pyyaml`
  * `einops` (tensor rearrangements)
  * `datasets` (Hugging Face datasets for front-loading tasks)
  * `wandb` or `tensorboard` for logging
  * *LTC reference implementation* (link in repo or install local module) — this project expects an LTC layer implementation in `models/layers/ltc.py`.

Example environment install (pip):

```bash
python -m venv venv
source venv/bin/activate
pip install torch numpy einops pyyaml tqdm datasets wandb
# add LTC lib if published or clone the LTC reference and install as editable
```

---

# 7. Starter hyperparameters (reference)

> **Perceiver**

* N\_lat = 512
* D\_lat = 1024
* Latent\_self\_blocks = 6
* Latent\_attention\_heads = 16

> **HLP**

* D\_H = 1024
* planner\_layers = 2
* planner\_update\_interval K = 4 (update planner every 4 inference ticks)

> **LLW (LTC)**

* D\_L = 512
* n\_micro\_steps = 16 (micro-steps per planner step)
* time\_step = 0.05 (simulated dt per micro-step) — tune based on task
* LTC cell initializer: small-time-constants bias (avoid extremely large τ)

> **Optimization**

* Optimizer: AdamW
* LR: 3e-4 (warmup 2k steps, linear decay)
* Batch size: small experiments 16–64 depending on compute
* Grad clip: 1.0
* Weight decay: 0.01
* Mixed precision: recommended but monitor LTC numerics

These are intentionally conservative; scale up/down as you experiment.

---

# 8. Training recipes & losses

**Primary loss**: task dependent (cross-entropy for language, MSE for regression, binary/softmax for classification).

**Auxiliary losses** (encouraged):

* Iterative consistency: supervise intermediate outputs or apply L2 penalty between successive HRM outputs when teacher data exists.
* Perceiver reconstruction: small autoencoding loss on a sampled subset of inputs to encourage latent fidelity.
* LLW temporal prediction: predict short-horizon next-step to stabilize LTC dynamics.
* LTC stability regularizer: L2 penalty on learned time-constants or bounded-scaling penalty.

**Curriculum & schedule**:

1. Start with teacher-forced planner & worker (drive LLW with ground-truth plans if possible).
2. Move to scheduled sampling: gradually replace teacher plan tokens with model outputs.
3. Anneal to free-running HRM where planner and worker operate solely on model-internal signals.

**Checkpointing & evaluation**: save checkpoints frequently (every epoch / N steps), and maintain validation roll-outs with both teacher-forced and free-running modes to detect divergence.

---

# 9. Recommended tasks & datasets (prioritized experiments)

1. **Synthetic iterative reasoning** (fast prototyping):

   * Sudoku generator (multiple difficulty levels), graph shortest-path with increasing graph size, small SAT instances.
   * Metrics: solve accuracy, steps-to-solution, sample efficiency.

2. **Long-context language**:

   * Datasets: PG-19, BookSum, Long Range Arena (text variants), NarrativeQA for long-document QA.
   * Metrics: perplexity / bits-per-token, QA F1/EM, summarization ROUGE.

3. **Multimodal VQA over long video**:

   * Datasets: YouTube-VIS style or long-video QA datasets.
   * Metrics: VQA accuracy, temporal grounding.

4. **Streaming sensor / event data**:

   * Event camera datasets, irregular-sampled clinical time-series (MIMIC-III vitals subset), telemetry anomaly datasets.
   * Metrics: detection AUC, latency-to-detection, RMSE for forecasting.

5. **Hierarchical control (simulator)**:

   * Environments: MuJoCo / Brax / PyBullet task requiring hierarchical goals (multi-stage pick & place, navigation+manipulation).
   * Metrics: success rate, smoothness, sample-efficiency.

Want a single concrete starter? Begin with **Sudoku**: it’s small, clearly measures hierarchical planning vs execution, and many baselines exist.

---

# 10. Ablation plan

Run a carefully controlled ablation matrix to validate claims:

1. Full model (Perceiver IO + HRM + LTC).
2. Perceiver IO only + standard decoder (no HRM, no LTC).
3. Perceiver IO + HRM where LLW is GRU (no LTC).
4. Perceiver IO + LLW(LTC) but no HLP (no hierarchical planner).
5. HRM + LTC but replace Perceiver with a Transformer encoder (to test importance of Perceiver's input scaling).

Measure sample efficiency (#samples to reach threshold), final accuracy, inference latency, and iteration stability.

---

# 11. Evaluation metrics (detailed)

* **Task accuracy**: problem-specific metric (accuracy, success rate, EM, BLEU, ROUGE, etc.).
* **Sample efficiency**: steps/samples to reach specified performance thresholds.
* **Iteration stability**: for iterative tasks, measure whether later iterations improve or degrade a validation metric (plot metric vs iteration).
* **Temporal robustness**: evaluate on irregularly subsampled versions of inputs; measure degradation.
* **Compute & latency**: mean latency per forward pass and memory envelope.
* **Interpretability / alignment**: correlation of planner tokens with human-level subgoals (requires human analysis).

Plot learning curves, iteration curves, and ablation bars.

---

# 12. Practical engineering & numerical pitfalls

* **LTC numerics**: LTC uses continuous dynamics; certain solvers or FP16 can destabilize training. Keep solver internals in FP32 if using AMP.
* **Gradient explosion**: use gradient clipping and norm penalties on recurrent weights.
* **Too many HRM iterations**: inference can be slow—limit iterations, or distill into a single-pass student model.
* **Representation mismatch**: if Perceiver latents are too small, HRM will starve; monitor latent capacity and activations.
* **Dataset bias & overfitting**: HRM can memorize small tasks — use strong held-out evaluation and regularization.

---

# 13. Codebase structure (suggested)

```
/ (root)
├─ README.md  # this file
├─ requirements.txt
├─ configs/   # YAML hyperparameter/config files for experiments
├─ models/
│  ├─ perceiver.py
│  ├─ hrm.py        # HLP + LLW glue
│  └─ layers/
│      ├─ ltc.py
│      ├─ attention.py
│      └─ utils.py
├─ data/
│  ├─ datasets/    # dataset drivers and preproc
├─ experiments/
│  ├─ sudoku_experiment.py
│  └─ long_context_narrative.py
├─ scripts/
│  ├─ train.py
│  └─ eval.py
└─ docs/
   └─ design_notes.md
```

---

# 14. Quickstart (development workflow)

1. Clone the repo and create the environment.

```bash
git clone <this-repo>
cd <this-repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run a tiny smoke test (toy data) — validates forward/backward pass:

```bash
python scripts/train.py --config configs/tiny_sudoku.yaml --dry_run True
```

3. Run a small training run (fast):

```bash
python scripts/train.py --config configs/sudoku_small.yaml
```

4. Evaluate and produce reports:

```bash
python scripts/eval.py --checkpoint runs/sudoku_small/checkpoint_last.pt --report_path results/sudoku_report.json
```

---

# 15. Reproducibility checklist

* [ ] Seed RNGs (torch, numpy, python) and log seeds.
* [ ] Save optimizer state and scheduler at each checkpoint.
* [ ] Use deterministic dataloaders where possible for debugging.
* [ ] Log complete config files with each run (save YAML).
* [ ] Keep validation & test splits fixed and documented.

---

# 16. Contribution guidelines

We welcome contributions. Suggested workflow:

1. Fork repo, code locally.
2. Run unit tests and linting.
3. Implement feature or experiment branch named `feat/<short-desc>` or `exp/<task>-<brief>`.
4. Open a PR with description, results, and a minimal reproduction script.

Include experimental results, hyperparams, and discussion in PR notes.

---

# 17. Expected outcomes & research hypotheses

This project tests the following hypotheses:

1. A hybrid Perceiver IO + HRM + LTC will improve sample efficiency on iterative symbolic or algorithmic reasoning tasks compared to Perceiver-only or Transformer baselines.
2. LTC-equipped low-level workers will provide superior robustness and accuracy on irregularly sampled and streaming temporal data compared to discrete-time RNN workers.
3. Perceiver latents + HRM planning will improve long-context performance (books, long videos) versus typical Transformers due to compressed workspace and iterative planning.

Each hypothesis is testable using the ablation plan and evaluation metrics included above.

---

# 18. Examples of future research directions

* Distillation: distill iterative HRM behavior into a single-pass student model for low-latency inference.
* Sparse routing in HLP (Mixture-of-Experts) for scaling planner capacity.
* Differentiable memory and retrieval-augmented HRM for long-term facts.
* Hardware-aware LTC solvers and neuromorphic-friendly variants for low-power edge deployment.

---

# 19. Licensing (MIT)

---

# 20. References (select canonical papers / resources — list for convenience)

* Perceiver / Perceiver IO — "Perceiver: General Perception with Iterative Attention" / "Perceiver IO" — Jaegle et al.
* Hierarchical Reasoning Model (HRM) — (user-provided) hierarchical multi-timescale reasoning preprint.
* Liquid Time-Constant Networks (LTC) — Hasani et al. / related continuous-time recurrent network papers.
* LTC-SE & continuous-time NN surveys — improvements and engineering notes.

@article{jaegle2021perceiver,
  title = {Perceiver: General Perception with Iterative Attention},
  author = {Jaegle, Andrew and Gimeno, Felix and Brock, Andrew and Zisserman, Andrew and Vinyals, Oriol and Carreira, Jo{\~a}o},
  year = {2021},
  journal = {arXiv},
  volume = {2103.03206},
  doi = {10.48550/arXiv.2103.03206},
  url = {https://arxiv.org/abs/2103.03206}
}

@article{jaegle2021perceiverio,
  title = {Perceiver IO: A General Architecture for Structured Inputs \& Outputs},
  author = {Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and H{\'e}naff, Olivier and Botvinick, Matthew M. and Zisserman, Andrew and Vinyals, Oriol and Carreira, Jo{\~a}o},
  year = {2021},
  journal = {arXiv},
  volume = {2107.14795},
  doi = {10.48550/arXiv.2107.14795},
  url = {https://arxiv.org/abs/2107.14795}
}

@article{hasani2020ltc_arxiv,
  title = {Liquid Time-constant Networks},
  author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Rus, Daniela and Grosu, Radu},
  year = {2020},
  journal = {arXiv},
  volume = {2006.04439},
  doi = {10.48550/arXiv.2006.04439},
  url = {https://arxiv.org/abs/2006.04439}
}

@inproceedings{hasani2021ltc_aaai,
  title = {Liquid Time-constant Networks},
  author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Rus, Daniela and Grosu, Radu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2021},
  volume = {35},
  pages = {7657--7666},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/16936}
}

@article{hasani2022closedform,
  title = {Closed-form Continuous-time Neural Networks},
  author = {Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and Teschl, Gerald and Rus, Daniela},
  journal = {Nature Machine Intelligence},
  year = {2022},
  volume = {4},
  number = {11},
  pages = {992--1003},
  doi = {10.1038/s42256-022-00556-7},
  url = {https://doi.org/10.1038/s42256-022-00556-7}
}

@article{bidollahkhani2023ltcse,
  title = {LTC-SE: Expanding the Potential of Liquid Time-Constant Neural Networks for Scalable AI and Embedded Systems},
  author = {Bidollahkhani, Michael and Atasoy, Ferhat and Abdellatef, Hamdan},
  year = {2023},
  journal = {arXiv},
  volume = {2304.08691},
  doi = {10.48550/arXiv.2304.08691},
  url = {https://arxiv.org/abs/2304.08691}
}

@article{wang2025hrm,
  title = {Hierarchical Reasoning Model (HRM)},
  author = {Wang, Guan and Li, Jin and Sun, Yuhao and Chen, Xing and Liu, Changling and Wu, Yue and Lu, Meng and Song, Sen and Yadkori, Yasin Abbasi},
  year = {2025},
  journal = {arXiv},
  volume = {2506.21734},
  doi = {10.48550/arXiv.2506.21734},
  url = {https://arxiv.org/abs/2506.21734}
}


---

# 21. Contact & support

If you use this work, open an issue or PR. For research collaboration, contact the maintainer(s) listed at the top of this file.

---

Thank you for reading.
