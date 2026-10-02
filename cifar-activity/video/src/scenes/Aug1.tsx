import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {lerp, prog, window} from '../lib/anim';
import {Abs, Arrow, Axes, DrawPath, Pix, Svg, Tag, Txt, linePath, scales} from '../components/ui';
import {R, aug} from '../lib/data';

const VIEWS = [
  {dy: 0, dx: 0, flip: false}, {dy: 4, dx: 4, flip: true}, {dy: 8, dx: 2, flip: false},
  {dy: 1, dx: 7, flip: true}, {dy: 6, dx: 8, flip: true}, {dy: 3, dx: 0, flip: false},
];
const NAMES: Record<string, string> = {
  none: 'none', flip: 'flip', crop: 'crop', 'crop-flip': 'crop + flip', 'color-jitter': '+ color jitter', rotation: '+ rotation',
  erasing: '+ erasing', mixup: '+ MixUp', cutmix: '+ CutMix', randaugment: '+ RandAugment', 'resized-crop': 'resized crop + flip',
  affine: '+ affine', grayscale: '+ grayscale', blur: '+ blur', autoaugment: '+ AutoAugment',
};

export const Aug1: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3, 4].map(seg);

  // Beat A/B: real horse, flip, pad, crop windows
  const viewO = window(f, s[0] - 6, s[2] + 4, 16);
  const flipP = Math.min(prog(f, word(0, 'Flip'), 16), 1 - prog(f, word(0, 'pad') - 4, 12));
  const padP = prog(f, word(0, 'pad'), 18);
  const PXS = 11.25; // screen px per image px
  const X0 = 200, Y0 = 330;
  const cropT = word(0, 'crop');
  const vi = Math.max(0, Math.min(5, Math.floor((f - cropT) / 18)));
  const mv = prog(f, cropT + vi * 18, 10);
  const prev = VIEWS[Math.max(0, vi - 1)], cur = VIEWS[vi];
  const wdx = f < cropT ? 4 : lerp(vi ? prev.dx : 4, cur.dx, mv);
  const wdy = f < cropT ? 4 : lerp(vi ? prev.dy : 4, cur.dy, mv);
  const invO = prog(f, s[1] + 4, 18);

  // Beat C: all 15 recipes
  const barO = window(f, s[2] - 2, s[3] + 4, 16);
  const recipes = [...R.augmentation.recipes].sort((a, b) => b.mean - a.mean);
  const B = {x: 560, y: 240, row: 36, lo: 55, hi: 72, w: 1000};
  const bx = (v: number) => B.x + ((v - B.lo) / (B.hi - B.lo)) * B.w;
  const hi: Record<string, [string, number]> = {
    none: [C.cream, word(2, 'No')], flip: [C.cyan, word(2, 'Flip')], 'crop-flip': [C.amber, word(2, 'Crop')],
  };

  // Beat D: train vs test curves
  const gapO = window(f, s[3] - 2, end + 20, 16);
  const curves = R.augmentation.curves_seed_mean;
  const gaps = R.augmentation.train_test_gap_pp;

  return (
    <AbsoluteFill>
      <Abs x={0} y={0} o={viewO}>
        {/* padded canvas */}
        <div style={{position: 'absolute', left: X0 - 4 * PXS, top: Y0 - 4 * PXS, width: 40 * PXS, height: 40 * PXS, background: '#000',
          opacity: padP, outline: `1px dashed ${C.dim}`}} />
        <Img src={staticFile('img/feat_horse.png')} style={{position: 'absolute', left: X0, top: Y0, width: 32 * PXS, height: 32 * PXS,
          imageRendering: 'pixelated', transform: `scaleX(${1 - 2 * flipP})`}} />
        <div style={{position: 'absolute', left: X0 + (wdx - 4) * PXS, top: Y0 + (wdy - 4) * PXS, width: 32 * PXS, height: 32 * PXS,
          border: `3px solid ${C.cyan}`, opacity: f > cropT - 4 ? 1 : 0, boxShadow: `0 0 16px ${C.cyan}`}} />
        <Txt x={X0 - 45} y={Y0 + 32 * PXS + 60} w={600} size={19} color={C.muted} font="mono" o={padP}>
          pad 4 black px → random 32×32 crop · flip p = 0.5
        </Txt>
        {VIEWS.map((v, i) => {
          const o = prog(f, cropT + i * 18 + 8, 10);
          return (
            <Pix key={i} src={`aug_horse_${i}.png`} x={830 + (i % 3) * 176} y={300 + Math.floor(i / 3) * 196} w={160} o={o}
              border={v.flip ? C.amber : C.line} />
          );
        })}
        <Txt x={830} y={700} w={640} size={18} color={C.muted} font="mono" o={prog(f, cropT + 40, 14)}>
          six training views of one real test image (amber = flipped)
        </Txt>
        <Svg o={invO}>
          {VIEWS.map((_, i) => (
            <Arrow key={i} x1={1350} y1={360 + Math.floor(i / 3) * 196 + (i % 3) * 40} x2={1500} y2={480}
              p={prog(f, s[1] + 4 + i * 3, 16)} color={C.cyanDim} width={1.5} />
          ))}
        </Svg>
        <Txt x={1520} y={446} w={360} size={56} font="serif" o={prog(f, s[1] + 22, 16)}>horse</Txt>
        <Txt x={1520} y={530} w={360} size={20} color={C.muted} o={prog(f, word(1, 'invariance'), 14)} lh={1.5}>
          same label for every view → <span style={{color: C.cyan}}>invariance</span>
        </Txt>
        <Txt x={1520} y={600} w={360} size={20} color={C.muted} o={prog(f, word(1, 'regularization') - 10, 14)} lh={1.5}>
          harder to memorize one exact image → <span style={{color: C.amber}}>regularization</span>
        </Txt>
      </Abs>

      <Abs x={0} y={0} o={barO}>
        <Tag x={B.x} y={190} kind="test">original 2-conv CNN · batch 512 · 80 epochs · lr 0.01 · 3 seeds · mean ± SD</Tag>
        <Svg>
          {[56, 60, 64, 68, 72].map((t) => (
            <g key={t}>
              <line x1={bx(t)} x2={bx(t)} y1={B.y - 6} y2={B.y + 15 * B.row} stroke={C.grid} />
              <text x={bx(t)} y={B.y + 15 * B.row + 26} fill={C.muted} fontFamily={F.mono} fontSize={16} textAnchor="middle">{t}%</text>
            </g>
          ))}
          {recipes.map((r, i) => {
            const y = B.y + i * B.row;
            const h = hi[r.recipe];
            const o = prog(f, s[2] + 6 + i * 2, 12);
            const lit = h ? prog(f, h[1] - 4, 10) : 0;
            const col = h ? h[0] : C.dim;
            const bo = h ? lerp(0.45, 1, lit) : 0.45;
            return (
              <g key={r.recipe} opacity={o}>
                <rect x={B.x} y={y + 6} width={bx(r.mean) - B.x} height={B.row - 12} fill={col} opacity={bo * 0.85} rx={2} />
                <line x1={bx(r.mean - r.sd)} x2={bx(r.mean + r.sd)} y1={y + B.row / 2} y2={y + B.row / 2} stroke={C.cream} strokeWidth={1.5} opacity={0.7} />
                <text x={B.x - 14} y={y + B.row / 2 + 6} fill={h ? col : C.muted} fontFamily={F.mono} fontSize={18} textAnchor="end">{NAMES[r.recipe]}</text>
                <text x={bx(r.mean + r.sd) + 10} y={y + B.row / 2 + 6} fill={h ? col : C.muted} fontFamily={F.mono} fontSize={17}>
                  {r.mean.toFixed(2)} ± {r.sd.toFixed(2)}
                </text>
              </g>
            );
          })}
        </Svg>
        <Txt x={B.x} y={B.y + 15 * B.row + 50} w={1100} size={17} font="mono" color={C.dim}>
          "+ X" recipes add X to crop + flip · ../runs/augmentation/comparison.json (45/45 runs)
        </Txt>
      </Abs>

      <Abs x={0} y={0} o={gapO}>
        {(['none', 'crop-flip'] as const).map((name, k) => {
          const b = {x: 260 + k * 820, y: 280, w: 600, h: 380};
          const {sx, sy} = scales(b.x, b.y, b.w, b.h, [1, 80], [20, 85]);
          const tr = curves[name].train, te = curves[name].test;
          const dp = prog(f, s[3] + 10 + k * 10, 50);
          const last = te.length;
          const col = k === 0 ? C.cream : C.amber;
          return (
            <React.Fragment key={name}>
              <Svg>
                <Axes x={b.x} y={b.y} w={b.w} h={b.h} xd={[1, 80]} yd={[20, 85]} xTicks={[1, 20, 40, 60, 80]} yTicks={[20, 40, 60, 80]}
                  xLabel="epoch" yLabel={k === 0 ? 'accuracy (%)' : undefined} />
                <rect x={sx(70)} y={sy(tr[last - 1])} width={sx(80) - sx(70)} height={Math.max(1, sy(te[last - 1]) - sy(tr[last - 1]))}
                  fill={C.amber} opacity={0.25 * prog(f, s[3] + 60, 16)} />
                <DrawPath d={linePath(tr.map((v, i) => [sx(i + 1), sy(v)]))} p={dp} stroke={C.muted} width={2} dash="5 5" />
                <DrawPath d={linePath(te.map((v, i) => [sx(i + 1), sy(v)]))} p={dp} stroke={col} width={3.5} />
              </Svg>
              <Txt x={b.x} y={b.y - 50} w={600} size={26} font="mono" color={col}>{NAMES[name]}</Txt>
              <Txt x={b.x + b.w - 300} y={b.y + 8} w={300} align="right" size={19} font="mono" color={C.muted} o={dp}>
                train (clean) {tr[last - 1].toFixed(1)}%<br />test {te[last - 1].toFixed(1)}%
              </Txt>
              <Txt x={b.x + b.w / 2} y={b.y + b.h + 64} w={500} align="center" size={30} font="serif" o={prog(f, s[3] + 60, 16)}>
                gap <span style={{color: C.amber}}>{gaps[name].toFixed(2)}</span> pp · test {aug(name).mean.toFixed(2)}%
              </Txt>
            </React.Fragment>
          );
        })}
        <Txt x={960} y={815} w={1500} align="center" size={30} font="serif" italic o={prog(f, word(3, 'useful') - 6, 16)}>
          Smaller gap: <span style={{color: C.cyan}}>useful regularization</span>… or <span style={{color: C.amber}}>a model that learned less</span>.
        </Txt>
        <Txt x={960} y={868} w={1500} align="center" size={20} font="mono" color={C.muted} o={prog(f, word(3, 'alone') - 4, 14)}>
          seed-mean curves · ../runs/augmentation/{'{none,crop-flip}'}/seed-*/run.log
        </Txt>
        <Tag x={960} y={168} kind="note" anchor="center" o={prog(f, s[4], 16)}>exploratory: this suite consulted the test set repeatedly</Tag>
      </Abs>
    </AbsoluteFill>
  );
};
