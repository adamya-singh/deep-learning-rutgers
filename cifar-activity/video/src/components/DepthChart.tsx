import React from 'react';
import {C, F} from '../theme';
import {Axes, DrawPath, linePath, scales} from './ui';
import {DEPTHS, dep} from '../lib/data';

export type DepthChartProps = {
  x: number; y: number; w: number; h: number;
  plain: number[]; // per-depth reveal 0..1 (index aligned with DEPTHS)
  residual?: number[]; // per-depth reveal for depths 4..32 (index aligned with DEPTHS)
  labels?: number;
  dimPlain?: number;
  yd?: [number, number];
};

// Per-point label offsets chosen so labels never sit on a connecting line.
const LABEL: Record<'plain' | 'residual', Record<number, [number, number]>> = {
  plain: {2: [14, 28], 4: [-14, -14], 8: [14, -14], 16: [14, -14], 32: [14, -14]},
  residual: {4: [14, 28], 8: [14, 28], 16: [-14, 28], 32: [14, 28]},
};

export const depthX = (x: number, w: number, d: number) => x + 40 + ((Math.log2(d) - 1) / 4) * (w - 80);

export const DepthChart: React.FC<DepthChartProps> = ({x, y, w, h, plain, residual = [0, 0, 0, 0, 0], labels = 1, dimPlain = 0, yd = [76, 89]}) => {
  const {sy} = scales(x, y, w, h, [0, 1], yd);
  const px = (d: number) => depthX(x, w, d);
  const series = (variant: 'plain' | 'residual', reveal: number[], color: string) => {
    const ds = DEPTHS.filter((d) => variant === 'plain' || d >= 4);
    const pts = ds.map((d) => [px(d), sy(dep(variant, d).mean)] as [number, number]);
    const idx = ds.map((d) => DEPTHS.indexOf(d));
    // Line grows segment by segment as points are revealed.
    let drawn = 0;
    for (let i = 1; i < ds.length; i++) drawn += Math.min(1, Math.max(0, reveal[idx[i]]));
    const p = ds.length > 1 ? drawn / (ds.length - 1) : 0;
    const o = variant === 'plain' ? 1 - 0.55 * dimPlain : 1;
    return (
      <g opacity={o}>
        <DrawPath d={linePath(pts)} p={p} stroke={color} width={3} />
        {ds.map((d, i) => {
          const c = dep(variant, d);
          const r = reveal[idx[i]];
          if (r <= 0) return null;
          return (
            <g key={d} opacity={Math.min(1, r * 1.5)}>
              <line x1={px(d)} x2={px(d)} y1={sy(c.mean - c.sd)} y2={sy(c.mean + c.sd)} stroke={color} strokeWidth={2} />
              <line x1={px(d) - 7} x2={px(d) + 7} y1={sy(c.mean - c.sd)} y2={sy(c.mean - c.sd)} stroke={color} strokeWidth={2} />
              <line x1={px(d) - 7} x2={px(d) + 7} y1={sy(c.mean + c.sd)} y2={sy(c.mean + c.sd)} stroke={color} strokeWidth={2} />
              <circle cx={px(d)} cy={sy(c.mean)} r={8} fill={C.bg} stroke={color} strokeWidth={3} />
              <text x={px(d) + LABEL[variant][d][0]} y={sy(c.mean) + LABEL[variant][d][1]} textAnchor={LABEL[variant][d][0] < 0 ? 'end' : 'start'} fill={color} fontFamily={F.mono} fontSize={20} opacity={labels}>{c.mean.toFixed(2)}</text>
            </g>
          );
        })}
      </g>
    );
  };
  return (
    <g>
      <Axes x={x} y={y} w={w} h={h} xd={[0, 1]} yd={yd} xTicks={DEPTHS} yTicks={[78, 80, 82, 84, 86, 88].filter((t) => t >= yd[0] && t <= yd[1])}
        xPos={px} xLabel="convolution layers" yLabel="validation accuracy (%)" yFmt={(v) => `${v}`} />
      {series('plain', plain, C.cream)}
      {series('residual', residual, C.cyan)}
    </g>
  );
};
