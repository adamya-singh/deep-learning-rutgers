import React from 'react';
import {C, F} from '../theme';
import {Slab} from './ui';

type Item =
  | {kind: 'slab'; x: number; size: number; depth: number; layers: number; label: string; dims: string; img?: string; color: string}
  | {kind: 'vec'; x: number; h: number; label: string; dims: string; color: string};

const ITEMS: Item[] = [
  {kind: 'slab', x: 0, size: 128, depth: 10, layers: 3, label: 'input', dims: '3×32×32', img: 'feat_frog.png', color: C.cream},
  {kind: 'slab', x: 230, size: 128, depth: 46, layers: 8, label: 'conv 3→32, ReLU', dims: '32×32×32', img: 'act1_13.png', color: C.cyan},
  {kind: 'slab', x: 480, size: 64, depth: 46, layers: 8, label: 'max pool', dims: '32×16×16', img: 'pool1_13.png', color: C.cyan},
  {kind: 'slab', x: 680, size: 64, depth: 80, layers: 12, label: 'conv 32→64, ReLU', dims: '64×16×16', img: 'act2_61.png', color: C.blue},
  {kind: 'slab', x: 920, size: 32, depth: 80, layers: 12, label: 'max pool', dims: '64×8×8', img: 'pool2_61.png', color: C.blue},
  {kind: 'vec', x: 1130, h: 230, label: 'flatten', dims: '4096', color: C.muted},
  {kind: 'vec', x: 1290, h: 150, label: 'dense, ReLU', dims: '128', color: C.cyan},
  {kind: 'vec', x: 1450, h: 80, label: 'logits', dims: '10', color: C.amber},
];
export const STRIP_WIDTH = 1520;
const BASE = 200; // y of the slab baseline (bottom of front face) inside the strip

export const ArchStrip: React.FC<{x: number; y: number; scale: number; active: number[]; reveal: number; labels?: number}> = ({
  x, y, scale, active, reveal, labels = 1,
}) => (
  <div style={{position: 'absolute', left: x, top: y, transform: `scale(${scale})`, transformOrigin: 'left top', width: STRIP_WIDTH, height: 320}}>
    {ITEMS.map((it, i) => {
      const shown = Math.max(0, Math.min(1, reveal * ITEMS.length - i));
      const on = active.length === 0 || active.includes(i);
      const o = shown * (on ? 1 : 0.32);
      if (it.kind === 'slab') {
        return (
          <div key={i} style={{position: 'absolute', left: it.x, top: BASE - it.size, opacity: o}}>
            <Slab x={0} y={0} size={it.size} depth={it.depth} layers={it.layers} color={it.color} img={it.img} glow={on && active.length ? 0.35 : 0} />
            <div style={{position: 'absolute', top: it.size + 16, left: -40, width: it.size + it.depth + 80, textAlign: 'center', opacity: labels}}>
              <div style={{fontFamily: F.sans, fontSize: 20, color: C.cream}}>{it.label}</div>
              <div style={{fontFamily: F.mono, fontSize: 18, color: C.muted, marginTop: 4}}>{it.dims}</div>
            </div>
          </div>
        );
      }
      const n = it.dims === '10' ? 10 : 14;
      return (
        <div key={i} style={{position: 'absolute', left: it.x, top: BASE - it.h, opacity: o}}>
          <div style={{width: 26, height: it.h, border: `1.5px solid ${it.color}`, borderRadius: 13, display: 'flex', flexDirection: 'column',
            justifyContent: 'space-evenly', alignItems: 'center', boxShadow: on && active.length ? `0 0 14px ${it.color}` : undefined}}>
            {Array.from({length: n}).map((_, k) => (
              <div key={k} style={{width: 8, height: 8, borderRadius: 4, background: it.color, opacity: 0.75}} />
            ))}
          </div>
          <div style={{position: 'absolute', top: it.h + 16, left: -60, width: 146, textAlign: 'center', opacity: labels}}>
            <div style={{fontFamily: F.sans, fontSize: 20, color: C.cream}}>{it.label}</div>
            <div style={{fontFamily: F.mono, fontSize: 18, color: C.muted, marginTop: 4}}>{it.dims}</div>
          </div>
        </div>
      );
    })}
    <svg width={STRIP_WIDTH} height={320} style={{position: 'absolute', left: 0, top: 0, overflow: 'visible'}}>
      {[[150, 222], [400, 472], [600, 672], [820, 912], [1040, 1118], [1166, 1280], [1326, 1440]].map(([a, b], i) => (
        <g key={i} opacity={Math.max(0, Math.min(1, reveal * ITEMS.length - i - 1)) * 0.8}>
          <line x1={a} y1={BASE - 40} x2={b - 8} y2={BASE - 40} stroke={C.dim} strokeWidth={2} />
          <path d={`M${b - 16},${BASE - 46} L${b - 8},${BASE - 40} L${b - 16},${BASE - 34}`} stroke={C.dim} strokeWidth={2} fill="none" />
        </g>
      ))}
    </svg>
  </div>
);
