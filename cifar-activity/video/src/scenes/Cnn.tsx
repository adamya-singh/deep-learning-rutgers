import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import {C, CLASSES, F} from '../theme';
import {useCues} from '../lib/time';
import {clamp, lerp, linear, prog, rand, window} from '../lib/anim';
import {Abs, Arrow, Eq, Frac, Pix, Slab, Sub, Svg, Tag, Txt, V} from '../components/ui';
import {ArchStrip} from '../components/ArchStrip';
import {S} from '../lib/data';

const MEAN = [0.4914, 0.4822, 0.4465];
const STD = [0.247, 0.2435, 0.2616];
const CH = ['R', 'G', 'B'];
const CHC = ['#ff8f7a', '#86e39b', '#86a9ff'];
const TOP1 = [13, 10, 28, 27, 22, 17, 24, 25];
const TOP2 = [61, 41, 42, 46, 19, 49, 39, 25, 37, 58, 54, 29, 45, 9, 26, 4];

const Cell: React.FC<{v: number; kind: 'x' | 'w'; size: number; o?: number}> = ({v, kind, size, o = 1}) => {
  const bg =
    kind === 'x'
      ? `rgba(243,233,214,${clamp(0.08 + Math.abs(v) * 0.22, 0, 0.5)})`
      : v >= 0
        ? `rgba(95,211,238,${clamp(Math.abs(v) * 2.2, 0.08, 0.7)})`
        : `rgba(233,166,64,${clamp(Math.abs(v) * 2.2, 0.08, 0.7)})`;
  return (
    <div style={{width: size, height: size, background: bg, border: `1px solid ${C.line}`, fontFamily: F.mono, fontSize: 15,
      color: C.cream, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: o}}>
      {Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2)}
    </div>
  );
};
const Grid3: React.FC<{vals: number[][]; kind: 'x' | 'w'; x: number; y: number; o?: number}> = ({vals, kind, x, y, o = 1}) => (
  <div style={{position: 'absolute', left: x, top: y, display: 'grid', gridTemplateColumns: 'repeat(3, 46px)', opacity: o}}>
    {vals.flat().map((v, i) => <Cell key={i} v={v} kind={kind} size={46} />)}
  </div>
);

export const Cnn: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const [s0, s1, s2, s3, s4, s5, s6, s7] = [0, 1, 2, 3, 4, 5, 6, 7].map(seg);

  // Architecture strip: big overview, then a compact persistent map at the top.
  const shrink = prog(f, s1, 32);
  const stripActive =
    f < s1 ? [] : f < s2 ? [0] : f < s5 ? [1] : f < s6 ? [2] : f < s7 ? [3, 4] : [5, 6, 7];
  const ms = S.multiply_sum;

  // --- normalization beat
  const nO = window(f, s1 + 22, s2 + 6);
  const [pr, pg, pb] = S.hero_pixels[ms.row][ms.col];
  const raw = [pr, pg, pb];

  // --- convolution beat
  const FX = 120, FY = 330, FS = 448, P = FS / 32;
  const convO = window(f, s2 + 4, s4 + 4, 18);
  const tSweep = word(2, 'Slide');
  const sweep = linear(f, tSweep, 120);
  const approach = prog(f, s2 + 6, 34);
  let wr: number, wc: number;
  if (f < tSweep) {
    wr = lerp(1, ms.row, approach);
    wc = lerp(1, ms.col, approach);
  } else {
    const idx = Math.min(1023, Math.floor(sweep * 1024));
    wr = Math.floor(idx / 32);
    wc = idx % 32;
  }
  const revealRows = f < tSweep ? 0 : sweep * 32;
  const panelO = window(f, s2 + 40, s3 + 4, 16);
  const partial = [0, 1, 2].map((c) => {
    let s = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += ms.patch[c][i][j] * ms.weights[c][i][j];
    return s;
  });
  const rowT = (c: number) => s2 + 52 + c * 22;
  const totalO = prog(f, s2 + 130, 18);
  const sharedO = window(f, s3 + 6, s4 + 4, 16);
  const SHARED: [number, number][] = [[5, 6], [9, 25], [23, 9], [26, 26]];

  // --- filters beat
  const filtO = window(f, s4 + 10, s5 + 4, 18);
  // --- relu / pool beat
  const poolO = window(f, s5 + 8, s6 + 4, 16);
  const EX = [[0.8, -0.3, 0.1, -1.2], [0.2, 1.4, -0.5, 0.3], [-0.9, 0.4, 0.7, 0.0], [0.6, -0.2, -1.1, 0.9]];
  const relu = prog(f, s5 + 50, 20);
  const pooled = prog(f, word(5, 'max') + 6, 20);
  // --- second block beat
  const b2O = window(f, s6 + 8, s7 + 4, 16);
  // --- flatten beat
  const flO = window(f, s7 + 8, end + 20, 16);
  const unroll = prog(f, s7 + 14, 40);

  return (
    <AbsoluteFill>
      <ArchStrip x={lerp(200, 600, shrink)} y={lerp(330, 92, shrink)} scale={lerp(1, 0.47, shrink)} active={stripActive}
        reveal={prog(f, s0 - 6, 60)} labels={1 - shrink} />
      <Txt x={960} y={lerp(640, 236, shrink)} w={900} align="center" size={lerp(40, 22, shrink)} font="serif"
        o={prog(f, word(0, '545,098') - 4, 16) * (1 - prog(f, s2 - 10, 14))}>
        <span style={{color: C.amber}}>545,098</span> trainable parameters
      </Txt>
      <Txt x={960} y={700} w={1300} align="center" size={20} font="mono" color={C.muted} o={window(f, word(0, '545,098') + 14, s1 + 10)}>
        conv1 896 · conv2 18,496 · dense 524,416 · logits 1,290 · source: ../cifar_cnn.py build_model()
      </Txt>

      {/* Normalization */}
      <Abs x={0} y={0} o={nO}>
        <Eq x={960} y={330} size={56}>
          <V>x̂</V><Sub>c</Sub> = <Frac n={<span><V>x</V><Sub>c</Sub> − <V c={C.cyan}>μ</V><Sub>c</Sub></span>} d={<span><V c={C.amber}>σ</V><Sub>c</Sub></span>} />
        </Eq>
        {[0, 1, 2].map((c) => (
          <div key={c} style={{position: 'absolute', left: 560 + c * 290, top: 520, width: 260, textAlign: 'center', fontFamily: F.mono, fontSize: 24,
            color: C.cream, opacity: prog(f, s1 + 40 + c * 8, 14)}}>
            <div style={{color: CHC[c], fontSize: 22, marginBottom: 8}}>{CH[c]}</div>
            <div><span style={{color: C.cyan}}>μ</span> = {MEAN[c].toFixed(4)}</div>
            <div><span style={{color: C.amber}}>σ</span> = {STD[c].toFixed(4)}</div>
            <div style={{color: C.muted, fontSize: 19, marginTop: 14}}>
              {raw[c]}/255 → {(((raw[c] / 255) - MEAN[c]) / STD[c]).toFixed(3)}
            </div>
          </div>
        ))}
        <Txt x={960} y={720} w={1100} align="center" size={20} color={C.muted} font="mono" o={prog(f, s1 + 80, 16)}>
          frog pixel (row {ms.row}, col {ms.col}) · CIFAR-10 training-set channel statistics from ../cifar_cnn.py
        </Txt>
      </Abs>

      {/* Convolution: real image, real filter, real numbers */}
      <Abs x={0} y={0} o={convO}>
        <Pix src="feat_frog.png" x={FX} y={FY} w={FS} />
        <div style={{position: 'absolute', left: FX + (wc - 1) * P, top: FY + (wr - 1) * P, width: 3 * P, height: 3 * P,
          border: `3px solid ${C.cyan}`, boxShadow: `0 0 18px ${C.cyan}`, opacity: 1 - sharedO}} />
        <Txt x={FX} y={FY + FS + 14} w={FS} size={18} color={C.muted} font="mono">input (normalized), 32×32×3</Txt>

        {/* Output map revealed behind the sweep */}
        <div style={{position: 'absolute', left: 1352, top: FY, width: FS, height: FS, border: `1px solid ${C.line}`}} />
        <Img src={staticFile(`img/act1_${ms.filter}.png`)} style={{position: 'absolute', left: 1352, top: FY, width: FS, height: FS,
          imageRendering: 'pixelated', clipPath: `inset(0 0 ${100 - (Math.floor(revealRows) / 32) * 100}% 0)`}} />
        <Img src={staticFile(`img/act1_${ms.filter}.png`)} style={{position: 'absolute', left: 1352, top: FY, width: FS, height: FS,
          imageRendering: 'pixelated',
          clipPath: `inset(${(Math.floor(revealRows) / 32) * 100}% ${100 - (revealRows % 1) * 100}% ${100 - ((Math.floor(revealRows) + 1) / 32) * 100}% 0)`,
          opacity: f >= tSweep && revealRows < 32 ? 1 : 0}} />
        <div style={{position: 'absolute', left: 1352 + ms.col * P, top: FY + ms.row * P, width: P, height: P, border: `2px solid ${C.amber}`,
          opacity: panelO}} />
        <Txt x={1352} y={FY + FS + 14} w={FS} size={18} color={C.muted} font="mono">output map, filter {ms.filter} (after ReLU)</Txt>
        <Tag x={1352} y={FY - 40} kind="measured" o={1}>baseline checkpoint, epoch 10</Tag>

        {/* Multiply-sum panel */}
        <Abs x={0} y={0} o={panelO}>
          <Txt x={620} y={FY - 6} w={700} size={22} color={C.cream}>
            one position · row {ms.row}, col {ms.col} · 27 multiplies, then a sum
          </Txt>
          {[0, 1, 2].map((c) => (
            <Abs key={c} x={0} y={0} o={prog(f, rowT(c), 14)}>
              <Txt x={612} y={FY + 82 + c * 146} w={40} size={22} color={CHC[c]} font="mono">{CH[c]}</Txt>
              <Grid3 vals={ms.patch[c]} kind="x" x={640} y={FY + 26 + c * 146} />
              <Txt x={792} y={FY + 76 + c * 146} w={40} size={30} color={C.muted} align="left">×</Txt>
              <Grid3 vals={ms.weights[c]} kind="w" x={826} y={FY + 26 + c * 146} />
              <Txt x={986} y={FY + 82 + c * 146} w={200} size={22} color={C.cream} font="mono">→ {partial[c] >= 0 ? '+' : ''}{partial[c].toFixed(3)}</Txt>
            </Abs>
          ))}
          <Txt x={640} y={FY + 476} w={700} size={24} font="mono" o={totalO}>
            Σ + b ({ms.bias.toFixed(3)}) = <span style={{color: C.cyan}}>{ms.pre_relu.toFixed(3)}</span> → ReLU → {Math.max(0, ms.pre_relu).toFixed(3)}
          </Txt>
          <Txt x={640} y={FY + 516} w={700} size={17} color={C.muted} font="mono" o={totalO}>
            real conv1 filter {ms.filter} weights · model's own output: {ms.check_from_model.toFixed(4)}
          </Txt>
        </Abs>

        {/* Shared weights */}
        <Abs x={0} y={0} o={sharedO}>
          <Txt x={960} y={FY - 4} w={760} align="center" size={24} font="serif">
            one filter: <span style={{color: C.cyan}}>27 weights + 1 bias</span>, reused at all 1,024 positions
          </Txt>
          <Pix src={`filter_${ms.filter}.png`} x={900} y={500} w={120} border={C.cyan} />
          <Txt x={960} y={632} w={400} align="center" size={18} color={C.muted} font="mono">filter {ms.filter} (3×3, RGB-scaled)</Txt>
          <Txt x={960} y={700} w={600} align="center" size={20} color={C.muted} font="mono" o={prog(f, s3 + 50, 16)}>
            32 filters × 28 = 896 parameters
          </Txt>
        </Abs>
        <Svg o={sharedO}>
          {SHARED.map(([r, c], i) => {
            const p = prog(f, s3 + 10 + i * 6, 18);
            const x0 = FX + (c - 1) * P, y0 = FY + (r - 1) * P;
            return (
              <g key={i} opacity={p}>
                <rect x={x0} y={y0} width={3 * P} height={3 * P} fill="none" stroke={C.cyan} strokeWidth={2.5} />
                <line x1={x0 + 3 * P} y1={y0 + 1.5 * P} x2={900} y2={560} stroke={C.cyan} strokeWidth={1.2} opacity={0.7} />
                <line x1={1020} y1={560} x2={1352 + c * P + P / 2} y2={FY + r * P + P / 2} stroke={C.cyan} strokeWidth={1.2} opacity={0.7} />
                <rect x={1352 + c * P} y={FY + r * P} width={P} height={P} fill="none" stroke={C.amber} strokeWidth={2} />
              </g>
            );
          })}
        </Svg>
      </Abs>

      {/* 32 learned filters and their maps */}
      <Abs x={0} y={0} o={filtO}>
        <Txt x={130} y={318} w={640} size={22}>32 learned filters</Txt>
        {Array.from({length: 32}).map((_, k) => (
          <Pix key={k} src={`filter_${k}.png`} x={130 + (k % 8) * 78} y={366 + Math.floor(k / 8) * 78} w={64}
            o={prog(f, s4 + 14 + k * 1.2, 10)} border={C.line} />
        ))}
        <Txt x={820} y={318} w={980} size={22}>their 32 feature maps for this frog</Txt>
        {Array.from({length: 32}).map((_, k) => (
          <Pix key={k} src={`act1_${k}.png`} x={820 + (k % 8) * 124} y={366 + Math.floor(k / 8) * 118} w={112}
            o={prog(f, s4 + 30 + k * 1.2, 10)} border={C.line} />
        ))}
        <Txt x={130} y={700} w={600} size={30} font="serif" italic color={C.amber} o={prog(f, word(4, 'Nobody'), 16)}>
          Not hand-designed.<br />Learned by gradient descent.
        </Txt>
        <Tag x={130} y={845} kind="measured" o={prog(f, s4 + 40, 16)}>conv1 weights + activations extracted from ../runs/baseline/last.pt (tools/extract_static.py)</Tag>
      </Abs>

      {/* ReLU and max pooling */}
      <Abs x={0} y={0} o={poolO}>
        <Eq x={960} y={300} size={48}>ReLU(<V>z</V>) = max(0, <V>z</V>)</Eq>
        {EX.map((row, i) =>
          row.map((v, j) => {
            const shown = v < 0 ? lerp(v, 0, relu) : v;
            const inPool = Math.floor(i / 2) * 2 + Math.floor(j / 2);
            const blockMax = Math.max(...[0, 1].flatMap((a) => [0, 1].map((b) => Math.max(0, EX[Math.floor(i / 2) * 2 + a][Math.floor(j / 2) * 2 + b]))));
            const isMax = Math.max(0, v) === blockMax;
            return (
              <div key={`${i}-${j}`} style={{position: 'absolute', left: 170 + j * 78, top: 420 + i * 78, width: 74, height: 74,
                border: `1px solid ${C.line}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: 22,
                color: v < 0 ? lerp(0, 1, relu) > 0.5 ? C.amber : C.cream : C.cream,
                background: pooled > 0 && isMax ? `rgba(95,211,238,${0.35 * pooled})` : ['rgba(36,52,94,0.25)', 'rgba(36,52,94,0.5)'][inPool % 2]}}>
                {shown.toFixed(1)}
              </div>
            );
          }),
        )}
        <Svg><Arrow x1={500} y1={576} x2={590} y2={576} p={pooled} color={C.cyan} /></Svg>
        {[0, 1].map((i) => [0, 1].map((j) => {
          const v = Math.max(...[0, 1].flatMap((a) => [0, 1].map((b) => Math.max(0, EX[i * 2 + a][j * 2 + b]))));
          return (
            <div key={`${i}${j}`} style={{position: 'absolute', left: 610 + j * 78, top: 498 + i * 78, width: 74, height: 74, border: `1px solid ${C.cyan}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: F.mono, fontSize: 22, color: C.cream, opacity: pooled}}>
              {v.toFixed(1)}
            </div>
          );
        }))}
        <Tag x={170} y={760} kind="illustrative" o={prog(f, s5 + 20, 14)}>example numbers</Tag>
        <Pix src={`act1_${ms.filter}.png`} x={1010} y={420} w={300} border={C.line} />
        <Svg><Arrow x1={1330} y1={570} x2={1420} y2={570} p={pooled} color={C.cyan} /></Svg>
        <Pix src={`pool1_${ms.filter}.png`} x={1440} y={420} w={300} border={C.cyan} o={pooled} />
        <Txt x={1160} y={735} w={300} align="center" size={20} font="mono" color={C.muted}>32 × 32</Txt>
        <Txt x={1590} y={735} w={300} align="center" size={20} font="mono" color={C.muted} o={pooled}>16 × 16</Txt>
        <Tag x={1010} y={378} kind="measured">real map, before and after 2×2 max pool</Tag>
      </Abs>

      {/* Second block */}
      <Abs x={0} y={0} o={b2O}>
        <Slab x={190} y={430} size={190} depth={90} layers={10} img={`pool1_${ms.filter}.png`} color={C.cyan}
          label="32 × 16 × 16" sub="pooled conv1 maps" />
        <Svg>
          <Arrow x1={520} y1={520} x2={640} y2={520} p={prog(f, s6 + 20, 16)} color={C.blue} />
          <Arrow x1={980} y1={520} x2={1090} y2={520} p={prog(f, s6 + 40, 16)} color={C.blue} />
        </Svg>
        <Slab x={700} y={470} size={90} depth={120} layers={14} color={C.blue} o={prog(f, s6 + 26, 14)} />
        <Txt x={790} y={640} w={420} align="center" size={20} font="mono" o={prog(f, s6 + 30, 14)}>
          one conv2 filter: 3×3×<span style={{color: C.cyan}}>32</span>
        </Txt>
        <Txt x={790} y={672} w={460} align="center" size={18} font="mono" color={C.muted} o={prog(f, s6 + 36, 14)}>
          288 weights + 1 bias · ×64 = 18,496
        </Txt>
        {TOP2.map((k, i) => (
          <Pix key={k} src={`act2_${k}.png`} x={1110 + (i % 4) * 172} y={330 + Math.floor(i / 4) * 124} w={112}
            o={prog(f, s6 + 44 + i * 1.5, 10)} border={C.line} />
        ))}
        {TOP2.slice(0, 16).map((k, i) => (
          <Pix key={`p${k}`} src={`pool2_${k}.png`} x={1110 + (i % 4) * 172 + 118} y={330 + Math.floor(i / 4) * 124 + 56} w={52}
            o={prog(f, word(6, 'Pooling') + i, 10)} border={C.blue} />
        ))}
        <Txt x={1110} y={830} w={700} size={18} font="mono" color={C.muted} o={prog(f, s6 + 60, 14)}>
          16 of 64 real conv2 maps (16×16) and pooled 8×8 versions
        </Txt>
      </Abs>

      {/* Flatten → dense → logits */}
      <Abs x={0} y={0} o={flO}>
        <Slab x={150} y={460} size={150} depth={120} layers={12} img="pool2_61.png" color={C.blue} label="64 × 8 × 8" o={1 - unroll * 0.6} />
        <Svg>
          {(() => {
            const cols = [
              {x: 700, n: 40, y0: 300, y1: 840, c: C.muted},
              {x: 1060, n: 16, y0: 390, y1: 750, c: C.cyan},
              {x: 1400, n: 10, y0: 360, y1: 790, c: C.amber},
            ];
            const yy = (k: number, i: number) => cols[k].y0 + ((cols[k].y1 - cols[k].y0) * i) / (cols[k].n - 1);
            const lines: React.ReactNode[] = [];
            for (let a = 0; a < 40; a += 2) for (let b = 0; b < 16; b++)
              lines.push(<line key={`a${a}-${b}`} x1={700} y1={yy(0, a)} x2={1060} y2={yy(1, b)} stroke={C.line} strokeWidth={0.6} opacity={0.5 * unroll} />);
            for (let a = 0; a < 16; a++) for (let b = 0; b < 10; b++)
              lines.push(<line key={`b${a}-${b}`} x1={1060} y1={yy(1, a)} x2={1400} y2={yy(2, b)} stroke={C.line} strokeWidth={0.8} opacity={0.7 * unroll} />);
            const parts: React.ReactNode[] = [];
            for (let i = 0; i < 46; i++) {
              const t = ((f - s7) * 0.012 + rand(i)) % 1;
              if (f < s7 + 50) continue;
              const a = Math.floor(rand(i + 7) * 40), b = Math.floor(rand(i + 13) * 16), c = Math.floor(rand(i + 29) * 10);
              const [x, y] = t < 0.5
                ? [lerp(700, 1060, t * 2), lerp(yy(0, a), yy(1, b), t * 2)]
                : [lerp(1060, 1400, t * 2 - 1), lerp(yy(1, b), yy(2, c), t * 2 - 1)];
              parts.push(<circle key={`p${i}`} cx={x} cy={y} r={3.2} fill={t < 0.5 ? C.cyan : C.amber} opacity={0.9} />);
            }
            return (
              <g>
                {lines}
                {cols.map((col, k) =>
                  Array.from({length: col.n}).map((_, i) => (
                    <circle key={`${k}-${i}`} cx={col.x} cy={yy(k, i)} r={k === 0 ? 4 : 8} fill={C.bg} stroke={col.c} strokeWidth={2}
                      opacity={k === 0 ? unroll : prog(f, s7 + 30 + k * 16 + i, 10)} />
                  )),
                )}
                {parts}
              </g>
            );
          })()}
        </Svg>
        <Txt x={700} y={252} w={260} align="center" size={22} font="mono" o={unroll}>4,096</Txt>
        <Txt x={1060} y={340} w={260} align="center" size={22} font="mono" o={prog(f, s7 + 46, 12)}>128 (16 shown)</Txt>
        <Txt x={1400} y={310} w={260} align="center" size={22} font="mono" o={prog(f, s7 + 60, 12)}>10 logits</Txt>
        {CLASSES.map((c, i) => (
          <Txt key={c} x={1426} y={346 + (430 * i) / 9} w={200} size={20} font="mono" color={i === 6 ? C.amber : C.muted}
            o={prog(f, s7 + 64 + i * 2, 10)}>{c}</Txt>
        ))}
        <Txt x={1060} y={866} w={1100} align="center" size={20} font="mono" color={C.muted} o={prog(f, s7 + 90, 16)}>
          dense: 4,096 × 128 + 128 = 524,416 parameters (96% of the network)
        </Txt>
      </Abs>
    </AbsoluteFill>
  );
};
