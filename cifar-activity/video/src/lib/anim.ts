import {Easing, interpolate} from 'remotion';

const smooth = Easing.bezier(0.45, 0, 0.2, 1);

/** 0→1 between start and start+dur (frames), eased. */
export const prog = (f: number, start: number, dur = 20, easing = smooth) =>
  interpolate(f, [start, start + dur], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing,
  });

export const linear = (f: number, start: number, dur: number) =>
  interpolate(f, [start, start + dur], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Visible from `start` (fade in) until `end` (fade out). */
export const window = (f: number, start: number, end: number, d = 14) =>
  Math.min(prog(f, start, d), 1 - prog(f, end - d, d));

export const clamp = (x: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));

/** Deterministic pseudo-random in [0,1). */
export const rand = (i: number) => {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

export const fmt = (x: number, d = 2) => x.toFixed(d);
export const commas = (x: number) => Math.round(x).toLocaleString('en-US');
