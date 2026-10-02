# Sources for the film

Primary local evidence: `experiment-evidence.md`, `results-snapshot.json`, and the parent training code and `runs/*` artifacts. The code uses torchvision 0.22.1; use its actual architecture rather than assuming the current docs version changes our implementation.

- CIFAR-10 official dataset description: https://www.cs.toronto.edu/~kriz/cifar.html
- Original residual-network paper: https://arxiv.org/abs/1512.03385
- Torchvision ResNet18 reference: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Wide Residual Networks (evidence that ResNet18 is not a universal best): https://arxiv.org/abs/1605.07146
- Remotion coding-agent workflow: https://www.remotion.dev/docs/ai/coding-agents
- Requested model ID, confirmed in Anthropic documentation: https://platform.claude.com/docs/en/models/opus-5-5/overview (`claude-opus-5-5`)

The film should describe measured results and mechanisms. It must not claim that our CIFAR-adapted ResNet18 is the best ResNet architecture in existence, or that architecture changes alone caused the observed recipe improvement.

## Local evidence used by the film (read-only)

- `experiment-evidence.md`, `results-snapshot.json` (brief inputs)
- Source code: `../cifar_cnn.py` (classroom model, normalization), `../depth_cnn.py` (plain/residual Block, zero-pad shortcut, LR schedule, split), `../followup_cnn.py` (CIFAR ResNet-18: 3×3 stem, no max pool, weights=None), `../augmentations.py` (pad-4 crop, flip p=0.5)
- Checkpoint: `../runs/baseline/last.pt` (filters, activations, logits, loss slice, all CPU and read-only)
- Logs/metrics: `../runs/{baseline,tuned-b512,step-matched-b512}.log`, `../runs/augmentation/*/seed-*/run.log`, `../runs/depth/*/conv-*/seed-*/metrics.jsonl`, `../runs/followup/*/seed-*/metrics.jsonl`
- Aggregates: `../runs/{augmentation,depth,followup}/comparison.json`, `../runs/followup/queue-state.json`, `../runs/followup/resnet18-crop-flip/seed-*/test-summary.json` (frozen copies in `data/frozen/`)
- Dataset: `../data/cifar-10-batches-py`
