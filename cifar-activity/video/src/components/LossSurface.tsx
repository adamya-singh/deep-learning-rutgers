import React from 'react';
import {C} from '../theme';
import {S} from '../lib/data';

const L = S.loss_slice.loss.map((row) => row.map((v) => Math.log(v)));
const N = L.length;
const ZMIN = Math.min(...L.flat());
const ZMAX = Math.max(...L.flat());

/** Bilinear log-loss at (a, b) in [-1, 1]^2. */
export const logLossAt = (a: number, b: number) => {
  const u = ((a + 1) / 2) * (N - 1);
  const v = ((b + 1) / 2) * (N - 1);
  const i = Math.max(0, Math.min(N - 2, Math.floor(u)));
  const j = Math.max(0, Math.min(N - 2, Math.floor(v)));
  const fu = u - i, fv = v - j;
  return L[i][j] * (1 - fu) * (1 - fv) + L[i + 1][j] * fu * (1 - fv) + L[i][j + 1] * (1 - fu) * fv + L[i + 1][j + 1] * fu * fv;
};

/** Gradient descent on the measured slice (illustrative trajectory, not recorded training). */
export const descentPath = (start: [number, number], lr: number, steps: number) => {
  const pts: [number, number][] = [start];
  let [a, b] = start;
  const e = 1e-3;
  for (let k = 0; k < steps; k++) {
    const ga = (logLossAt(a + e, b) - logLossAt(a - e, b)) / (2 * e);
    const gb = (logLossAt(a, b + e) - logLossAt(a, b - e)) / (2 * e);
    a = Math.max(-1, Math.min(1, a - lr * ga));
    b = Math.max(-1, Math.min(1, b - lr * gb));
    pts.push([a, b]);
  }
  return pts;
};

export type View = {cx: number; cy: number; r: number; hgt: number; yaw: number; pitch: number};

export const project = (a: number, b: number, z: number, v: View): [number, number] => {
  const X = a * v.r, Y = b * v.r;
  const Xr = X * Math.cos(v.yaw) - Y * Math.sin(v.yaw);
  const Yr = X * Math.sin(v.yaw) + Y * Math.cos(v.yaw);
  const zn = (z - ZMIN) / (ZMAX - ZMIN);
  return [v.cx + Xr, v.cy + Yr * Math.sin(v.pitch) - zn * v.hgt * Math.cos(v.pitch)];
};

const mix = (t: number) => {
  // cyan (low) → amber (high)
  const c1 = [95, 211, 238], c2 = [233, 166, 64];
  const k = Math.max(0, Math.min(1, t));
  return `rgb(${c1.map((c, i) => Math.round(c + (c2[i] - c) * k)).join(',')})`;
};

export const LossSurface: React.FC<{view: View; draw: number; o?: number}> = ({view, draw, o = 1}) => {
  const lines: React.ReactNode[] = [];
  const g = (k: number) => -1 + (2 * k) / (N - 1);
  const shown = Math.floor(draw * N);
  for (let i = 0; i < N; i++) {
    if (i > shown) break;
    for (const dir of [0, 1]) {
      const pts: string[] = [];
      let zsum = 0;
      for (let j = 0; j < N; j++) {
        const [a, b] = dir === 0 ? [g(i), g(j)] : [g(j), g(i)];
        const z = dir === 0 ? L[i][j] : L[j][i];
        zsum += z;
        const [x, y] = project(a, b, z, view);
        pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
      }
      const t = (zsum / N - ZMIN) / (ZMAX - ZMIN);
      lines.push(<polyline key={`${dir}-${i}`} points={pts.join(' ')} fill="none" stroke={mix(t * 1.4)} strokeWidth={1.3} opacity={0.75} />);
    }
  }
  return <g opacity={o}>{lines}</g>;
};
