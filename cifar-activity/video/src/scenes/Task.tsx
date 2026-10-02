import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import {C, CLASSES, F} from '../theme';
import {useCues} from '../lib/time';
import {lerp, prog} from '../lib/anim';
import {Abs, Pix, Svg, Tag, Txt} from '../components/ui';
import {S} from '../lib/data';

const G = {x: 350, y: 168, cell: 58, pitch: 66};

export const Task: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s1 = seg(1);
  const s2 = seg(2);
  const s3 = seg(3);

  // Hero frog: center → gallery slot → big pixel view.
  const frogRow = 6;
  const toGallery = prog(f, s1, 26);
  const toBig = prog(f, s2, 30);
  const A = {x: 760, y: 250, s: 400};
  const B = {x: G.x, y: G.y + frogRow * G.pitch, s: G.cell};
  const Cc = {x: 150, y: 170, s: 640};
  const fx = toBig > 0 ? lerp(B.x, Cc.x, toBig) : lerp(A.x, B.x, toGallery);
  const fy = toBig > 0 ? lerp(B.y, Cc.y, toBig) : lerp(A.y, B.y, toGallery);
  const fs = toBig > 0 ? lerp(B.s, Cc.s, toBig) : lerp(A.s, B.s, toGallery);

  const galleryO = Math.min(prog(f, s1 + 8, 20), 1 - prog(f, s2, 20));
  const introO = prog(f, 8, 24) * (1 - prog(f, s1, 14));

  // Pixel view.
  const gridO = prog(f, s2 + 30, 24);
  const r0 = 13, c0 = 14, k = 4; // inspected 4x4 block (row, col)
  const insetO = Math.min(prog(f, s2 + 50, 24), 1 - prog(f, word(2, '32', 2) - 6, 18));
  const px = Cc.s / 32;
  const tSplit = word(2, '32', 2);
  const split = prog(f, tSplit, 36);
  const slabO = prog(f, tSplit - 10, 20);
  const countO = prog(f, word(2, '3,072'), 18);
  const qO = prog(f, s3 + 4, 24);

  return (
    <AbsoluteFill>
      {/* Intro caption under the hero frog */}
      <Txt x={960} y={672} w={900} align="center" size={22} color={C.muted} o={introO} font="mono">
        CIFAR-10 test image #{S.featured.frog.test_index} · label: frog · 32 × 32 pixels
      </Txt>

      {/* Gallery: first 8 test images of every class */}
      <Abs x={0} y={0} o={galleryO}>
        {CLASSES.map((name, r) => (
          <React.Fragment key={name}>
            <Txt x={G.x - 20} y={G.y + r * G.pitch + 16} w={220} align="right" size={21} color={C.muted} font="mono"
              o={prog(f, s1 + 10 + r * 3, 14)}>
              {name}
            </Txt>
            {Array.from({length: 8}).map((_, c) =>
              r === frogRow && c === 0 ? null : (
                <Pix key={c} src={`cifar_${r}_${c}.png`} x={G.x + c * G.pitch} y={G.y + r * G.pitch} w={G.cell}
                  o={prog(f, s1 + 10 + r * 3 + c * 1.5, 12)} />
              ),
            )}
          </React.Fragment>
        ))}
        <Abs x={1010} y={250} w={780}>
          <div style={{fontFamily: F.serif, fontSize: 92, color: C.cream, opacity: prog(f, word(1, '60,000') - 4, 16)}}>60,000</div>
          <div style={{fontFamily: F.sans, fontSize: 26, color: C.muted, marginTop: 4, opacity: prog(f, word(1, '60,000') + 8, 16)}}>
            tiny color photos · 32 × 32 · RGB · 10 classes
          </div>
        </Abs>
        <Svg>
          {(() => {
            const p = prog(f, word(1, '50,000'), 30);
            const pt = prog(f, word(1, '10,000'), 24);
            const x = 1010, y = 470, w = 760;
            return (
              <g>
                <rect x={x} y={y} width={w * (50 / 60) * p} height={48} fill={C.cyan} opacity={0.85} rx={3} />
                <rect x={x + w * (50 / 60) + 4} y={y} width={(w * (10 / 60) - 4) * pt} height={48} fill={C.amber} opacity={0.9} rx={3} />
                <text x={x} y={y + 86} fill={C.cyan} fontFamily={F.mono} fontSize={24} opacity={p}>50,000 train</text>
                <text x={x + w} y={y + 86} fill={C.amber} fontFamily={F.mono} fontSize={24} textAnchor="end" opacity={pt}>10,000 test</text>
              </g>
            );
          })()}
        </Svg>
        <Tag x={1010} y={600} kind="note" o={prog(f, word(1, '10,000') + 10, 16)}>official split · cs.toronto.edu/~kriz/cifar.html</Tag>
      </Abs>

      {/* The frog itself (persistent object) */}
      <Img src={staticFile('img/feat_frog.png')}
        style={{position: 'absolute', left: fx, top: fy, width: fs, height: fs, imageRendering: 'pixelated',
          opacity: Math.min(prog(f, 0, 18), 1 - 0.75 * split * 0)}} />

      {/* Pixel grid + inspected block */}
      <Svg o={gridO}>
        {Array.from({length: 33}).map((_, i) => (
          <g key={i}>
            <line x1={Cc.x + i * px} x2={Cc.x + i * px} y1={Cc.y} y2={Cc.y + Cc.s} stroke="rgba(6,12,29,0.55)" strokeWidth={1} />
            <line y1={Cc.y + i * px} y2={Cc.y + i * px} x1={Cc.x} x2={Cc.x + Cc.s} stroke="rgba(6,12,29,0.55)" strokeWidth={1} />
          </g>
        ))}
        <rect x={Cc.x + c0 * px} y={Cc.y + r0 * px} width={k * px} height={k * px} fill="none" stroke={C.amber} strokeWidth={3} opacity={insetO} />
        <g opacity={insetO} stroke={C.amber} strokeWidth={1} strokeDasharray="4 6">
          <line x1={Cc.x + (c0 + k) * px} y1={Cc.y + r0 * px} x2={900} y2={200} />
          <line x1={Cc.x + (c0 + k) * px} y1={Cc.y + (r0 + k) * px} x2={900} y2={200 + 4 * 112} />
        </g>
      </Svg>
      <Abs x={900} y={200} o={insetO}>
        {Array.from({length: k * k}).map((_, i) => {
          const r = Math.floor(i / k), c = i % k;
          const [R, Gr, B] = S.hero_pixels[r0 + r][c0 + c];
          const appear = prog(f, s2 + 60 + i * 2, 10);
          return (
            <div key={i} style={{position: 'absolute', left: c * 112, top: r * 112, width: 108, height: 108, background: `rgb(${R},${Gr},${B})`,
              opacity: appear, fontFamily: F.mono, fontSize: 19, lineHeight: '30px', paddingTop: 10, textAlign: 'center',
              color: R + Gr + B > 380 ? '#0a0f20' : C.cream}}>
              <div style={{color: '#ffb3a6'}}>{R}</div>
              <div style={{color: '#b8f5c4'}}>{Gr}</div>
              <div style={{color: '#b9d2ff'}}>{B}</div>
            </div>
          );
        })}
        <Txt x={0} y={4 * 112 + 14} w={460} size={21} color={C.muted} font="mono" o={prog(f, s2 + 90, 16)}>
          real pixel values, rows {r0}–{r0 + k - 1}, cols {c0}–{c0 + k - 1} (R, G, B · 0–255)
        </Txt>
      </Abs>

      {/* R, G, B slabs */}
      <div style={{position: 'absolute', left: 1080, top: 230, width: 600, height: 420, perspective: 1500, opacity: slabO}}>
        {(['B', 'G', 'R'] as const).map((ch, i) => {
          const depth = (2 - i) * 120 * split;
          return (
            <div key={ch} style={{position: 'absolute', left: 120 + depth * 0.55, top: 40 - depth * 0.28, width: 330, height: 330,
              transform: `rotateY(-28deg) rotateX(8deg)`, transformOrigin: 'left center'}}>
              <Img src={staticFile(`img/frog_${ch}.png`)} style={{width: 330, height: 330, imageRendering: 'pixelated', opacity: 0.95,
                outline: `1.5px solid ${ch === 'R' ? '#ff8f7a' : ch === 'G' ? '#86e39b' : '#86a9ff'}`}} />
              <div style={{position: 'absolute', top: -34, left: 0, fontFamily: F.mono, fontSize: 22,
                color: ch === 'R' ? '#ff8f7a' : ch === 'G' ? '#86e39b' : '#86a9ff', opacity: split}}>{ch}</div>
            </div>
          );
        })}
      </div>
      <Txt x={1380} y={690} w={760} align="center" size={34} font="serif" o={countO}>
        3 × 32 × 32 = <span style={{color: C.amber}}>3,072</span> numbers
      </Txt>

      <Txt x={960} y={835} w={1500} align="center" size={42} font="serif" italic o={qO * (1 - prog(f, end - 10, 10))}>
        What makes a network better at turning these numbers into the right answer?
      </Txt>
    </AbsoluteFill>
  );
};
