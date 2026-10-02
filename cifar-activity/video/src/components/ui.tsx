import React from 'react';
import {Img, staticFile} from 'remotion';
import {C, F, H, W} from '../theme';

type Box = {x: number; y: number; w?: number; h?: number};

export const Abs: React.FC<Box & {style?: React.CSSProperties; children?: React.ReactNode; o?: number}> = ({
  x, y, w, h, style, children, o = 1,
}) => (
  <div style={{position: 'absolute', left: x, top: y, width: w, height: h, opacity: o, ...style}}>{children}</div>
);

export const Txt: React.FC<{
  x: number; y: number; w?: number; size?: number; color?: string; font?: 'sans' | 'serif' | 'mono';
  weight?: number; italic?: boolean; align?: 'left' | 'center' | 'right'; o?: number; children: React.ReactNode;
  style?: React.CSSProperties; lh?: number;
}> = ({x, y, w, size = 28, color = C.cream, font = 'sans', weight = 400, italic, align = 'left', o = 1, children, style, lh = 1.3}) => {
  // For centered/right text, x is the anchor; w is the box width.
  const width = w ?? 1200;
  const left = align === 'center' ? x - width / 2 : align === 'right' ? x - width : x;
  return (
    <div
      style={{
        position: 'absolute', left, top: y, width, fontFamily: F[font], fontSize: size, color, fontWeight: weight,
        fontStyle: italic ? 'italic' : 'normal', textAlign: align, opacity: o, lineHeight: lh, ...style,
      }}
    >
      {children}
    </div>
  );
};

/** Small evidence label: what kind of thing the viewer is looking at. */
export const Tag: React.FC<{x: number; y: number; kind: 'measured' | 'schematic' | 'illustrative' | 'test' | 'validation' | 'note' | 'code'; o?: number; children: React.ReactNode; anchor?: 'left' | 'right' | 'center'}> = ({
  x, y, kind, o = 1, children, anchor = 'left',
}) => {
  const col = {
    measured: C.cyan, schematic: C.amber, illustrative: C.amber, test: C.amber, validation: C.cyan, note: C.muted, code: C.muted,
  }[kind];
  const label = {measured: 'MEASURED', schematic: 'SCHEMATIC', illustrative: 'ILLUSTRATIVE', test: 'TEST', validation: 'VALIDATION', note: 'NOTE', code: 'SOURCE'}[kind];
  const tx = anchor === 'right' ? 'translateX(-100%)' : anchor === 'center' ? 'translateX(-50%)' : undefined;
  return (
    <div
      style={{
        position: 'absolute', left: x, top: y, opacity: o, transform: tx, display: 'flex', alignItems: 'center', gap: 10,
        fontFamily: F.mono, fontSize: 15, color: C.muted, whiteSpace: 'nowrap',
      }}
    >
      <span style={{border: `1px solid ${col}`, color: col, padding: '3px 8px', borderRadius: 4, letterSpacing: 1.5, fontSize: 13}}>{label}</span>
      <span>{children}</span>
    </div>
  );
};

/** A CIFAR-sized image shown with crisp, visible pixels. */
export const Pix: React.FC<Box & {src: string; o?: number; style?: React.CSSProperties; border?: string}> = ({x, y, w, h, src, o = 1, style, border}) => (
  <Img
    src={staticFile(`img/${src}`)}
    style={{
      position: 'absolute', left: x, top: y, width: w, height: h ?? w, imageRendering: 'pixelated', opacity: o,
      outline: border ? `1px solid ${border}` : undefined, ...style,
    }}
  />
);

/** Full-frame SVG overlay for lines, charts and diagrams. */
export const Svg: React.FC<{children: React.ReactNode; o?: number}> = ({children, o = 1}) => (
  <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{position: 'absolute', left: 0, top: 0, opacity: o, overflow: 'visible'}}>
    {children}
  </svg>
);

const hash = (s: string) => {
  let x = 0;
  for (let i = 0; i < s.length; i++) x = (x * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(x).toString(36);
};

/** Path that draws itself from 0 → p (dashed paths are revealed through a mask). */
export const DrawPath: React.FC<{d: string; p: number; stroke?: string; width?: number; dash?: string; o?: number; fill?: string}> = ({
  d, p, stroke = C.cyan, width = 3, dash, o = 1, fill = 'none',
}) => {
  if (p <= 0) return null;
  if (!dash) {
    return (
      <path d={d} stroke={stroke} strokeWidth={width} fill={fill} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - p}
        strokeLinecap="round" strokeLinejoin="round" opacity={o} />
    );
  }
  const id = `m${hash(d + stroke)}`;
  return (
    <g opacity={o}>
      <mask id={id} maskUnits="userSpaceOnUse" x={0} y={0} width={W} height={H}>
        <path d={d} stroke="white" strokeWidth={width + 8} fill="none" pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - p} />
      </mask>
      <path d={d} stroke={stroke} strokeWidth={width} fill={fill} strokeDasharray={dash} mask={`url(#${id})`} />
    </g>
  );
};

export const Arrow: React.FC<{x1: number; y1: number; x2: number; y2: number; p?: number; color?: string; width?: number; head?: number; o?: number; dash?: string}> = ({
  x1, y1, x2, y2, p = 1, color = C.muted, width = 2, head = 10, o = 1, dash,
}) => {
  const ex = x1 + (x2 - x1) * p;
  const ey = y1 + (y2 - y1) * p;
  const a = Math.atan2(y2 - y1, x2 - x1);
  const hx1 = ex - head * Math.cos(a - 0.45);
  const hy1 = ey - head * Math.sin(a - 0.45);
  const hx2 = ex - head * Math.cos(a + 0.45);
  const hy2 = ey - head * Math.sin(a + 0.45);
  if (p <= 0) return null;
  return (
    <g opacity={o}>
      <line x1={x1} y1={y1} x2={ex} y2={ey} stroke={color} strokeWidth={width} strokeDasharray={dash} />
      <path d={`M${hx1},${hy1} L${ex},${ey} L${hx2},${hy2}`} stroke={color} strokeWidth={width} fill="none" strokeLinejoin="round" />
    </g>
  );
};

/** Equation line set in serif. Use <V> for italic variables. */
export const Eq: React.FC<{x: number; y: number; size?: number; o?: number; align?: 'left' | 'center'; children: React.ReactNode; color?: string}> = ({
  x, y, size = 46, o = 1, align = 'center', children, color = C.cream,
}) => (
  <div
    style={{
      position: 'absolute', left: align === 'center' ? x - 800 : x, top: y, width: align === 'center' ? 1600 : undefined,
      textAlign: align, fontFamily: F.serif, fontSize: size, color, opacity: o, whiteSpace: 'nowrap', lineHeight: 1.2,
    }}
  >
    {children}
  </div>
);
export const V: React.FC<{children: React.ReactNode; c?: string}> = ({children, c}) => (
  <span style={{fontStyle: 'italic', color: c}}>{children}</span>
);
export const Sub: React.FC<{children: React.ReactNode}> = ({children}) => (
  <sub style={{fontSize: '0.6em', verticalAlign: '-0.25em'}}>{children}</sub>
);
export const Sup: React.FC<{children: React.ReactNode}> = ({children}) => (
  <sup style={{fontSize: '0.6em', verticalAlign: '0.6em', lineHeight: 0}}>{children}</sup>
);
export const Frac: React.FC<{n: React.ReactNode; d: React.ReactNode}> = ({n, d}) => (
  <span style={{display: 'inline-flex', flexDirection: 'column', alignItems: 'center', verticalAlign: 'middle', margin: '0 0.2em'}}>
    <span style={{padding: '0 0.15em 0.08em'}}>{n}</span>
    <span style={{borderTop: `2px solid currentColor`, padding: '0.08em 0.15em 0', width: '100%', textAlign: 'center'}}>{d}</span>
  </span>
);

/** Scales for a chart box. */
export const scales = (x: number, y: number, w: number, h: number, xd: [number, number], yd: [number, number]) => ({
  sx: (v: number) => x + ((v - xd[0]) / (xd[1] - xd[0])) * w,
  sy: (v: number) => y + h - ((v - yd[0]) / (yd[1] - yd[0])) * h,
});

export const linePath = (pts: [number, number][]) => pts.map(([a, b], i) => `${i ? 'L' : 'M'}${a.toFixed(1)},${b.toFixed(1)}`).join(' ');

export const Axes: React.FC<{
  x: number; y: number; w: number; h: number; xd: [number, number]; yd: [number, number];
  xTicks: number[]; yTicks: number[]; xLabel?: string; yLabel?: string; o?: number;
  xFmt?: (v: number) => string; yFmt?: (v: number) => string; grid?: boolean; xLog?: boolean; xPos?: (v: number) => number;
}> = ({x, y, w, h, xd, yd, xTicks, yTicks, xLabel, yLabel, o = 1, xFmt = String, yFmt = String, grid = true, xPos}) => {
  const {sx, sy} = scales(x, y, w, h, xd, yd);
  const px = xPos ?? sx;
  return (
    <g opacity={o} fontFamily={F.mono} fontSize={16} fill={C.muted}>
      {grid && yTicks.map((t) => <line key={`g${t}`} x1={x} x2={x + w} y1={sy(t)} y2={sy(t)} stroke={C.grid} strokeWidth={1} />)}
      <line x1={x} x2={x + w} y1={y + h} y2={y + h} stroke={C.line} strokeWidth={1.5} />
      <line x1={x} x2={x} y1={y} y2={y + h} stroke={C.line} strokeWidth={1.5} />
      {xTicks.map((t) => (
        <g key={`x${t}`}>
          <line x1={px(t)} x2={px(t)} y1={y + h} y2={y + h + 6} stroke={C.line} />
          <text x={px(t)} y={y + h + 26} textAnchor="middle">{xFmt(t)}</text>
        </g>
      ))}
      {yTicks.map((t) => (
        <text key={`y${t}`} x={x - 12} y={sy(t) + 5} textAnchor="end">{yFmt(t)}</text>
      ))}
      {xLabel && <text x={x + w / 2} y={y + h + 58} textAnchor="middle" fill={C.muted} fontFamily={F.sans} fontSize={18}>{xLabel}</text>}
      {yLabel && (
        <text x={x - 70} y={y + h / 2} textAnchor="middle" fill={C.muted} fontFamily={F.sans} fontSize={18} transform={`rotate(-90 ${x - 70} ${y + h / 2})`}>
          {yLabel}
        </text>
      )}
    </g>
  );
};

/** A stack of feature maps seen slightly from the side. */
export const Slab: React.FC<{
  x: number; y: number; size: number; depth: number; layers?: number; color?: string; img?: string; o?: number; label?: string; sub?: string; glow?: number;
}> = ({x, y, size, depth, layers = 6, color = C.cyan, img, o = 1, label, sub, glow = 0}) => {
  const n = Math.max(1, layers);
  const step = n > 1 ? depth / (n - 1) : 0;
  return (
    <div style={{position: 'absolute', left: x, top: y, opacity: o}}>
      {Array.from({length: n}).map((_, i) => {
        const k = n - 1 - i; // back to front
        return (
          <div
            key={i}
            style={{
              position: 'absolute', left: k * step, top: -k * step * 0.6, width: size, height: size,
              border: `1.5px solid ${color}`, background: k === 0 ? 'rgba(8,14,32,0.9)' : 'rgba(8,14,32,0.82)',
              boxShadow: glow ? `0 0 ${24 * glow}px ${color}` : undefined, opacity: k === 0 ? 1 : 0.55 + 0.45 * (1 - k / n),
            }}
          />
        );
      })}
      {img && <Img src={staticFile(`img/${img}`)} style={{position: 'absolute', left: 0, top: 0, width: size, height: size, imageRendering: 'pixelated'}} />}
      {label && (
        <div style={{position: 'absolute', left: -60, width: size + depth + 120, top: size + 14, textAlign: 'center', fontFamily: F.mono, fontSize: 17, color: C.cream}}>
          {label}
          {sub && <div style={{fontSize: 14, color: C.muted, marginTop: 4, fontFamily: F.sans}}>{sub}</div>}
        </div>
      )}
    </div>
  );
};

export const Panel: React.FC<Box & {o?: number; children?: React.ReactNode; border?: string}> = ({x, y, w, h, o = 1, children, border = C.line}) => (
  <div style={{position: 'absolute', left: x, top: y, width: w, height: h, opacity: o, background: C.panel, border: `1px solid ${border}`, borderRadius: 10}}>
    {children}
  </div>
);

/** Clip children (chart series) to a plot box so out-of-domain values never spill over axes. */
export const Clip: React.FC<{x: number; y: number; w: number; h: number; children: React.ReactNode}> = ({x, y, w, h, children}) => {
  const id = `clip-${Math.round(x)}-${Math.round(y)}-${Math.round(w)}-${Math.round(h)}`;
  return (
    <g>
      <clipPath id={id}>
        <rect x={x} y={y - 4} width={w + 4} height={h + 4} />
      </clipPath>
      <g clipPath={`url(#${id})`}>{children}</g>
    </g>
  );
};

/** Typographic minus for negative numbers. */
export const signed = (v: number, d = 2) => (v < 0 ? `−${Math.abs(v).toFixed(d)}` : `+${v.toFixed(d)}`);
