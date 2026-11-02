# Spec: Training

**Status:** Draft. Implemented in M2. **Crate:** `polypous-train`

Training lets apps such as a visual builder train a [Graph IR](graph-ir.md) model from a
declarative configuration, with progress events and cancellation. The scope is small-to-medium
models on a single device. Fine-tuning large pretrained models is out of scope for v1.

## 1. How it works

- `polypous-backend` provides a `DynamicModel` that holds the graph's parameters as Burn `Param`s and
  **implements Burn's `Module` trait by hand**. Burn optimizers, gradient handling and device moves
  therefore work unchanged.
- The forward pass is the normal graph interpreter running on Burn's **autodiff backend** wrapper.
  Because Burn records operations at runtime, gradients work for runtime-defined graphs.
- The training device is chosen like inference: GPU if the model, optimizer state and a batch fit,
  else CPU. Training never uses disk streaming.

## 2. Configuration (JSON)

```json
{
  "format": "polypous.training",
  "format_version": 1,
  "seed": 42,
  "device": "auto",
  "epochs": 10,
  "batch_size": 64,

  "data": {
    "train":      { "kind": "image_folder", "path": "data/train", "image_size": [28, 28], "grayscale": true },
    "validation": { "kind": "split", "from": "train", "fraction": 0.1 },
    "fields": { "image": "image", "label": "label" },
    "shuffle": true
  },

  "inputs":  { "image": "image" },
  "loss":    { "kind": "cross_entropy", "prediction": "logits", "target": "label", "label_smoothing": 0.0 },
  "metrics": [ { "kind": "accuracy", "prediction": "logits", "target": "label" } ],

  "optimizer": { "kind": "adamw", "lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.999] },
  "schedule":  { "kind": "cosine", "warmup_steps": 100, "min_lr": 1e-5 },
  "gradient_clip": { "max_norm": 1.0 },

  "checkpoint":     { "dir": "runs/mnist", "every_epochs": 1, "keep_last": 3, "keep_best": "val.accuracy" },
  "early_stopping": { "metric": "val.loss", "mode": "min", "patience": 3 }
}
```

- `inputs` maps graph inputs to dataset fields. `loss`/`metrics` refer to graph outputs and dataset fields.
- Before training starts, the config is **validated against the graph**: fields exist, dtypes
  and shapes of prediction and target are compatible, and parameters are trainable. Errors are
  reported like graph validation errors.

## 3. Catalog (v1)

| Category | Kinds |
|---|---|
| Optimizers | `sgd{momentum, nesterov}`, `adam`, `adamw`, `rmsprop` |
| Schedules | `constant`, `step{every, gamma}`, `exponential`, `cosine{warmup_steps, min_lr}`, `linear_warmup` |
| Losses | `mse`, `l1`, `huber`, `cross_entropy{label_smoothing}`, `binary_cross_entropy{with_logits}` |
| Metrics | `loss`, `accuracy`, `top_k_accuracy`, `precision`, `recall`, `f1`, `mae`, `rmse` |
| Datasets | `tensors` (in-memory `TensorData` from the app), `safetensors` file, `csv{columns}`, `image_folder` (class per sub-folder), `split` |
| Augmentation (images) | `random_flip`, `random_crop`, `normalize{mean,std}` |

## 4. Events

`Started { device, steps_per_epoch }`, `EpochStarted { epoch }`,
`Step { epoch, step, loss, lr }` (throttled, at most ~10/s), `EpochFinished { epoch, metrics }`,
`CheckpointSaved { path }`, `EarlyStopped`, `Finished { best_metrics, elapsed }`.

Cancellation stops after the current step. The app chooses whether a final checkpoint is written.

## 5. Checkpoints and export

- Checkpoint directory: `graph.json`, `weights.safetensors`, `optimizer.safetensors`,
  `state.json` (epoch, step, schedule state, RNG state, metrics history, config). Training can resume from it exactly.
- `export` writes inference artifacts only: `*.graph.json` + `*.safetensors`.

## 6. Correctness tests

- The gradient of every differentiable op matches PyTorch autograd on fixtures.
- Optimizer steps match PyTorch for 3 steps from identical initial state.
- An end-to-end MNIST CNN reaches ≥ 98% test accuracy within 5 epochs.
- Resuming from a checkpoint produces bit-identical results to uninterrupted training (CPU, fixed seed).
