# Pretraining Readiness Plan

This document defines everything to complete before renting a GPU and
starting pretraining. It focuses on reliability, reproducibility, cost
control, and safe recovery from failures.

---

## 1) Goal

Prepare the project so training runs are:

- reproducible (same setup can be rerun),
- recoverable (resume after crash or disconnect),
- observable (clear logs and metrics),
- publishable (usable pretrained checkpoint for others),
- cost-controlled (clear stop criteria and budget limits).

---

## 2) Scope

This phase covers planning and implementation of:

1. Run configuration management
2. Checkpointing and resume support
3. Logging and evaluation outputs
4. Training safety guards
5. Local validation before cloud run
6. Cloud artifact persistence
7. Stopping criteria and budget guardrails

This phase does **not** include architecture redesign or dataset changes.

---

## 3) Deliverables

Before cloud pretraining, the repo should contain:

- A run config file format and per-run saved config artifact
- Checkpoint system (`latest.pt` and `best.pt`)
- Resume path (`--resume`)
- Structured logs (console + file)
- Periodic text generation snapshots
- Safety checks (NaN guard, grad clipping, interrupt save)
- Local verification checklist completed
- Separate inference loader script validated
- Cloud sync instructions for persistent artifacts
- Defined stop criteria (quality + budget + time)

---

## 4) Ordered Implementation Plan

### Step 1 - Establish run configuration

Define one canonical run configuration object/file that includes:

- Model hyperparameters: `n_embd`, `n_head`, `n_layer`, `block_size`,
  `dropout`
- Training hyperparameters: `batch_size`, `learning_rate`, `max_iters`,
  `eval_interval`, `eval_iters`
- Runtime settings: `seed`, `device`, precision mode (if added),
  gradient clip threshold
- Data settings: dataset location and train/val split
- Run metadata: run name, output directory, timestamp

Acceptance criteria:

- Run config can be printed at start of run
- Same config is saved into run artifact folder

---

### Step 2 - Add run artifact directory structure

Use a per-run structure similar to:

- `runs/<run_name>/checkpoints/`
- `runs/<run_name>/logs/`
- `runs/<run_name>/samples/`
- `runs/<run_name>/metrics/`
- `runs/<run_name>/config.json`

Acceptance criteria:

- Directories are created automatically at run start
- All outputs are written only inside the active run folder

---

### Step 3 - Implement checkpoint strategy (`latest.pt` + `best.pt`)

Checkpoint payload should include:

- `model_state_dict`
- `optimizer_state_dict`
- current global step
- best validation loss so far
- full run config
- tokenizer metadata (`stoi`, `itos`, vocab chars)

Policy:

- Save `latest.pt` at regular interval (overwrite)
- Save `best.pt` only when validation loss improves

Acceptance criteria:

- Both files are produced with expected update behavior
- Checkpoint content includes all required keys

---

### Step 4 - Add resume support (`--resume`)

Add a resume path parameter that:

- loads checkpoint file,
- restores model + optimizer states,
- restores training step and best val loss,
- continues training from next step.

Acceptance criteria:

- Interrupted run can continue without restarting from step 0
- Log clearly states resumed step and checkpoint path

---

### Step 5 - Add structured logging (console + file)

Log at least:

- step number
- train loss
- validation loss
- learning rate
- elapsed time
- throughput estimate (optional but recommended)

Output destinations:

- stdout console
- `runs/<run_name>/logs/train.log`
- optional machine-readable metrics file (CSV/JSONL)

Acceptance criteria:

- Logs are readable in real time and persisted on disk
- Metric values are easy to parse after run

---

### Step 6 - Add eval text snapshots

At each evaluation interval:

- generate fixed-length sample text from a fixed prompt or start token,
- save snapshot to
  `runs/<run_name>/samples/sample_step_<step>.txt`.

Acceptance criteria:

- A sample file is created every eval interval
- Snapshot filenames include step number

---

### Step 7 - Add safety protections

Implement:

- NaN/Inf loss guard (abort safely if invalid loss appears)
- gradient clipping (configurable max norm)
- graceful interrupt handling:
  - on `KeyboardInterrupt`, save emergency checkpoint then exit cleanly

Acceptance criteria:

- Invalid numeric state does not continue silently
- Interrupting run still preserves resumable state

---

### Step 8 - Build separate inference loader script

Create inference-only script that:

- loads checkpoint (`best.pt` or explicit path),
- reconstructs model from saved config,
- restores tokenizer metadata,
- generates text without training mode.

Acceptance criteria:

- Inference works in fresh process from saved checkpoint only
- No dependency on in-memory training objects

---

### Step 9 - Local dry-run verification

Run short local test (CPU is fine) and validate full lifecycle:

1. start train,
2. produce checkpoints,
3. stop,
4. resume from checkpoint,
5. finish short run,
6. run inference script with saved checkpoint.

Acceptance criteria:

- End-to-end loop completes with no manual patching
- Artifacts are complete and readable

---

### Step 10 - Vast.ai persistence and run operations

Define runbook for cloud execution:

- start training inside `tmux`/`screen`
- periodically sync `runs/<run_name>/` to persistent storage
- verify sync after each checkpoint cycle
- stop instance only after final sync confirmation

Recommended persistent target:

- Hugging Face Hub model repo or other durable storage

Acceptance criteria:

- Losing instance does not lose all progress
- Latest useful checkpoint is recoverable off-instance

---

### Step 11 - Define stop criteria before launch

Set explicit stopping rules:

- Max steps: hard cap
- Early stop rule: no meaningful val improvement after patience window
- Budget cap: max dollar spend for a run
- Time cap: max wall clock duration

Acceptance criteria:

- Training can be terminated by policy, not guesswork
- Budget overruns are prevented by predefined limits

---

## 5) Master TODO Checklist

### Core engineering TODOs

- [ ] Create run config file and save it per run
- [ ] Implement `latest.pt` + `best.pt` checkpoint strategy
- [ ] Save tokenizer metadata inside checkpoint
- [ ] Add `--resume` loading path
- [ ] Add structured logging (console + file)
- [ ] Add eval text snapshots every eval interval
- [ ] Add NaN guard + grad clipping
- [ ] Add graceful interrupt checkpoint save

### Validation TODOs

- [ ] Test full cycle locally: train -> save -> resume -> generate
- [ ] Validate checkpoint can be loaded in a separate inference script

### Cloud reliability TODOs

- [ ] Set up persistent artifact sync from Vast.ai
- [ ] Define stop criteria (max steps, early stop rule, budget cap)

---

## 6) Suggested execution order (strict)

1. Config + run directories
2. Checkpoint save/load + resume
3. Logging + metrics files
4. Eval sample snapshots
5. NaN guard + grad clipping + interrupt handler
6. Inference loader script
7. Local lifecycle validation
8. Vast.ai sync procedure
9. Finalize stop criteria and budget policy
10. Start cloud pretraining

---

## 7) Risks if skipped

- No checkpointing -> lost training progress on disconnect/crash
- No resume path -> wasted compute after interruption
- No saved config/tokenizer -> checkpoint cannot be reused correctly
- No logs -> impossible to debug divergence or compare runs
- No stop criteria -> uncontrolled spending and overtraining risk

---

## 8) Definition of Ready for Cloud Pretraining

Cloud pretraining can start only when all items below are true:

- [ ] All master TODO items are completed
- [ ] Local dry-run validation passed
- [ ] Inference from saved checkpoint confirmed
- [ ] Artifact sync path tested
- [ ] Budget and stop criteria documented and approved

