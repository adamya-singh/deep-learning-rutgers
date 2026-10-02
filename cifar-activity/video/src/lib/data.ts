// Frozen evidence. Regenerate with tools/extract_static.py and tools/extract_results.py.
import staticJson from '../../public/data/static.json';
import resultsJson from '../../public/data/results.json';

type Featured = {test_index: number; label: string; logits: number[]; probs: number[]};

export const S = staticJson as unknown as {
  gallery_test_indices: number[][];
  checkpoint: {path: string; epoch: number; steps: number; wandb_id: string};
  featured: Record<string, Featured>;
  hero_pixels: number[][][];
  conv1_weights: number[][][][];
  conv1_bias: number[];
  act1_mean: number[];
  act2_mean: number[];
  multiply_sum: {
    filter: number;
    row: number;
    col: number;
    patch: number[][][];
    weights: number[][][];
    bias: number;
    pre_relu: number;
    check_from_model: number;
  };
  loss_slice: {alpha: number[]; beta: number[]; loss: number[][]; method: string; center_loss: number};
  extracted_utc: string;
};

type CurvePoint = {epoch: number; loss: number; train: number; test: number};
export type DepthCandidate = {
  variant: 'plain' | 'residual';
  depth: number;
  mean: number;
  sd: number;
  params: number;
  val_curve: number[];
  train_curve: number[];
  seed_val_curves: number[][];
  final_train: number;
};
export type FollowupCandidate = {
  candidate: string;
  seeds: number[];
  mean: number;
  sd: number | null;
  paired_gain: number[];
  params: number;
  per_seed: {seed: number; final10: number; peak: number; final_epoch: number; epochs: number}[];
  val_curve: number[];
  train_curve: number[];
};

export const R = resultsJson as unknown as {
  snapshot_utc: string;
  classroom: {baseline_b64: CurvePoint[]; tuned_b512_10ep: CurvePoint[]; b512_80ep: CurvePoint[]};
  augmentation: {
    recipes: {recipe: string; mean: number; sd: number; delta_vs_none: number[]}[];
    curves_seed_mean: Record<string, {test: number[]; train: number[]}>;
    train_test_gap_pp: Record<string, number>;
  };
  depth: {
    candidates: DepthCandidate[];
    test_mean: number;
    test_sd: number;
    lr_schedule: number[];
    steps_per_epoch: number;
  };
  followup: {
    phase: string;
    jobs: Record<string, string>;
    selection: unknown;
    test_results_available: boolean;
    test_mean?: number;
    test_sd?: number | null;
    candidates: FollowupCandidate[];
    running_epochs: Record<string, number>;
  };
};

export const dep = (variant: 'plain' | 'residual', depth: number) =>
  R.depth.candidates.find((c) => c.variant === variant && c.depth === depth)!;
export const fu = (name: string) => R.followup.candidates.find((c) => c.candidate === name)!;
export const aug = (name: string) => R.augmentation.recipes.find((r) => r.recipe === name)!;
export const DEPTHS = [2, 4, 8, 16, 32];
