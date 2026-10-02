import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, CLASSES, F} from '../theme';
import {useCues} from '../lib/time';
import {lerp, linear, prog, rand, window} from '../lib/anim';
import {Abs, Arrow, Axes, DrawPath, Eq, Frac, Pix, Sub, Sup, Svg, Tag, Txt, V, linePath, scales} from '../components/ui';
import {LossSurface, descentPath, logLossAt, project} from '../components/LossSurface';
import {R, S} from '../lib/data';

const FROG = S.featured.frog;
const PATH = descentPath([0.82, -0.86], 0.035, 70);

export const Learn: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(seg);

  // ---- logits → probabilities (bars morph)
  const barsO = window(f, s[0] - 10, s[4] + 6, 16);
  const morph = prog(f, word(1, 'Exponentiate') + 20, 40);
  const ZERO = 610, UNIT = 38, PX = 360, PW = 560;
  const frogHi = prog(f, s[2], 16);
  const ceO = prog(f, s[3], 18);
  const p = FROG.probs[6];

  // ---- detail
  const detO = window(f, s[4] + 8, s[5] + 6, 16);
  // ---- backprop
  const bpO = window(f, s[5] + 4, s[6] + 6, 16);
  // ---- landscape
  const lsO = window(f, s[6] + 4, s[7] + 6, 16);
  const view = {cx: 700, cy: 640, r: 360, hgt: 330, yaw: lerp(-0.62, -0.32, linear(f, s[6], end - s[6])), pitch: 0.95};
  const pathP = linear(f, s[6] + 70, 200);
  const nShown = Math.max(1, Math.floor(pathP * (PATH.length - 1)) + 1);
  const pts = PATH.slice(0, nShown).map(([a, b]) => project(a, b, logLossAt(a, b) + 0.04, view));
  // ---- batches
  const epO = window(f, s[7] + 4, s[8] + 6, 16);
  const fill = linear(f, s[7] + 20, Math.max(30, s[8] - s[7] - 40));
  const steps = Math.round(fill * 782);
  // ---- baseline curve
  const blO = prog(f, s[8] + 4, 16);
  const curve = R.classroom.baseline_b64;
  const ch = {x: 380, y: 300, w: 1100, h: 470};
  const {sx, sy} = scales(ch.x, ch.y, ch.w, ch.h, [0, 10], [0, 100]);
  const drawP = prog(f, s[8] + 16, 70);

  // backprop network (schematic)
  const cols = [5, 7, 7, 6, 4];
  const nx = (k: number) => 420 + k * 270;
  const ny = (k: number, i: number) => 600 + (i - (cols[k] - 1) / 2) * 62;
  const fwd = linear(f, s[5] + 6, 50);
  const bwd = linear(f, s[5] + 70, 60);

  return (
    <AbsoluteFill>
      {/* Logit / probability bars */}
      <Abs x={0} y={0} o={barsO}>
        <Pix src="feat_frog.png" x={180} y={176} w={96} o={1} border={C.line} />
        <Txt x={292} y={196} w={600} size={22} color={C.muted} font="mono">
          {morph < 0.5 ? 'logits zᵢ (real model output)' : 'softmax probabilities pᵢ'}
        </Txt>
        <Txt x={292} y={228} w={600} size={17} color={C.dim} font="mono">baseline checkpoint · test image #{FROG.test_index}</Txt>
        {CLASSES.map((c, i) => {
          const z = FROG.logits[i];
          const pr = FROG.probs[i];
          const zl = Math.min(ZERO, ZERO + z * UNIT), zw = Math.abs(z) * UNIT;
          const pl = PX, pw = pr * PW;
          const left = lerp(zl, pl, morph), width = lerp(zw, pw, morph);
          const y = 300 + i * 52;
          const isFrog = i === 6;
          const col = isFrog ? lerp(0, 1, frogHi) > 0.5 ? C.cyan : z >= 0 ? C.blue : C.brown : z >= 0 && morph < 0.5 ? C.blue : morph >= 0.5 ? C.blue : C.brown;
          const o = prog(f, s[0] + i * 3, 12) * (frogHi > 0 && !isFrog ? lerp(1, 0.45, frogHi) : 1);
          return (
            <React.Fragment key={c}>
              <Txt x={170} y={y + 2} w={170} align="right" size={21} font="mono" color={isFrog ? C.cream : C.muted} o={o}>{c}</Txt>
              <div style={{position: 'absolute', left, top: y, width: Math.max(2, width), height: 32, background: col, opacity: o * 0.9, borderRadius: 2}} />
              <Txt x={left + width + 12} y={y + 2} w={200} size={20} font="mono" o={o}>
                {morph < 0.5 ? z.toFixed(2) : `${(pr * 100).toFixed(pr < 0.001 ? 3 : 1)}%`}
              </Txt>
            </React.Fragment>
          );
        })}
        <Svg>
          <line x1={lerp(ZERO, PX, morph)} x2={lerp(ZERO, PX, morph)} y1={290} y2={830} stroke={C.muted} strokeWidth={1.5} />
        </Svg>
        <Txt x={1004} y={300} w={760} size={26} color={C.cyan} font="serif" italic o={frogHi * (1 - ceO * 0)}>
          {' '}
        </Txt>
      </Abs>

      {/* Softmax + cross-entropy equations */}
      <Abs x={0} y={0} o={Math.min(prog(f, s[1] + 4, 20), 1 - prog(f, s[4] + 2, 14))}>
        <Eq x={1330} y={260} size={50}>
          <V>p</V><Sub><V>i</V></Sub> = <Frac n={<span><V>e</V><Sup><V>z</V><Sub>i</Sub></Sup></span>} d={<span>Σ<Sub><V>j</V></Sub> <V>e</V><Sup><V>z</V><Sub>j</Sub></Sup></span>} />
        </Eq>
        <Txt x={1330} y={410} w={700} align="center" size={20} color={C.muted} font="mono" o={prog(f, s[1] + 40, 16)}>
          positive · sums to 1
        </Txt>
        <Txt x={1330} y={460} w={700} align="center" size={30} font="serif" color={C.cyan} o={frogHi * (1 - ceO)}>
          frog: {(p * 100).toFixed(1)}% — most of the bet, not all of it
        </Txt>
      </Abs>
      <Abs x={0} y={0} o={Math.min(ceO, 1 - prog(f, s[4] + 2, 14))}>
        <Eq x={1330} y={455} size={48}>
          <V>L</V> = −log <V>p</V><Sub><V>y</V></Sub>
        </Eq>
        <Svg>
          {(() => {
            const b = {x: 1060, y: 560, w: 560, h: 250};
            const sc = scales(b.x, b.y, b.w, b.h, [0, 1], [0, 5]);
            const curvePts: [number, number][] = [];
            for (let k = 0; k <= 100; k++) {
              const pp = 0.0067 + (k / 100) * (1 - 0.0067);
              curvePts.push([sc.sx(pp), sc.sy(-Math.log(pp))]);
            }
            const dp = prog(f, s[3] + 30, 40);
            return (
              <g>
                <Axes x={b.x} y={b.y} w={b.w} h={b.h} xd={[0, 1]} yd={[0, 5]} xTicks={[0, 0.5, 1]} yTicks={[0, 2.5, 5]} grid={false}
                  xLabel="probability on the correct class" />
                <DrawPath d={linePath(curvePts)} p={dp} stroke={C.cream} width={2.5} />
                <circle cx={sc.sx(p)} cy={sc.sy(-Math.log(p))} r={9} fill={C.cyan} opacity={prog(f, word(3, 'Confident') - 4, 12)} />
                <text x={sc.sx(p) - 30} y={sc.sy(-Math.log(p)) - 34} fill={C.cyan} fontFamily={F.mono} fontSize={19} textAnchor="start"
                  opacity={prog(f, word(3, 'Confident'), 12)}>this frog: −log {p.toFixed(3)} = {(-Math.log(p)).toFixed(2)}</text>
                <circle cx={sc.sx(0.02)} cy={sc.sy(-Math.log(0.02))} r={9} fill={C.amber} opacity={prog(f, word(3, 'Confident', 1), 12)} />
                <text x={sc.sx(0.02) + 22} y={sc.sy(-Math.log(0.02)) + 6} fill={C.amber} fontFamily={F.mono} fontSize={19}
                  opacity={prog(f, word(3, 'Confident', 1), 12)}>confident & wrong: p = 0.02 → 3.91 (example)</text>
              </g>
            );
          })()}
        </Svg>
      </Abs>

      {/* PyTorch detail: the model emits logits; the loss applies log-softmax */}
      <Abs x={0} y={0} o={detO}>
        <Eq x={960} y={300} size={46}>
          CE(<V>z</V>, <V>y</V>) = −log softmax(<V>z</V>)<Sub><V>y</V></Sub> = −<V>z</V><Sub><V>y</V></Sub> + log Σ<Sub><V>j</V></Sub> <V>e</V><Sup><V>z</V><Sub>j</Sub></Sup>
        </Eq>
        <div style={{position: 'absolute', left: 430, top: 450, width: 1060, padding: '26px 34px', background: 'rgba(5,10,24,0.85)',
          border: `1px solid ${C.line}`, borderRadius: 10, fontFamily: F.mono, fontSize: 24, color: C.cream, lineHeight: 1.7}}>
          <div><span style={{color: C.muted}}># the model ends with</span> nn.Linear(128, 10) <span style={{color: C.muted}}># → logits</span></div>
          <div>logits = model(x)</div>
          <div style={{opacity: prog(f, word(4, "PyTorch's"), 14)}}>loss = nn.CrossEntropyLoss()(<span style={{color: C.amber}}>logits</span>, y)
            <span style={{color: C.muted}}>  # log-softmax inside</span></div>
        </div>
        <Tag x={430} y={690} kind="code" o={prog(f, s[4] + 30, 14)}>../cifar_cnn.py build_model() and training loop</Tag>
        <Txt x={960} y={750} w={1200} align="center" size={28} font="serif" italic color={C.muted} o={prog(f, word(4, 'Softmax'), 16)}>
          Softmax is how we read the outputs, not an extra layer.
        </Txt>
      </Abs>

      {/* Backprop schematic */}
      <Abs x={0} y={0} o={bpO}>
        <Eq x={960} y={250} size={46}>
          <Frac n={<span>∂<V>L</V></span>} d={<span>∂<V>w</V></span>} /> = <Frac n={<span>∂<V>L</V></span>} d={<span>∂<V>z</V></span>} /> · <Frac n={<span>∂<V>z</V></span>} d={<span>∂<V>w</V></span>} />
        </Eq>
        <Svg>
          {cols.slice(0, -1).map((n, k) =>
            Array.from({length: n}).map((_, i) =>
              Array.from({length: cols[k + 1]}).map((__, j) => (
                <line key={`${k}-${i}-${j}`} x1={nx(k)} y1={ny(k, i)} x2={nx(k + 1)} y2={ny(k + 1, j)} stroke={C.line} strokeWidth={1} />
              )),
            ),
          )}
          {Array.from({length: 40}).map((_, q) => {
            const k = Math.floor(rand(q) * 4);
            const i = Math.floor(rand(q + 3) * cols[k]), j = Math.floor(rand(q + 5) * cols[k + 1]);
            const fwdT = fwd * 4 - k;
            const bwdT = bwd * 4 - (3 - k);
            const out: React.ReactNode[] = [];
            if (fwdT > 0 && fwdT < 1) out.push(<circle key={`f${q}`} cx={lerp(nx(k), nx(k + 1), fwdT)} cy={lerp(ny(k, i), ny(k + 1, j), fwdT)} r={4} fill={C.cyan} />);
            if (bwdT > 0 && bwdT < 1) out.push(<circle key={`b${q}`} cx={lerp(nx(k + 1), nx(k), bwdT)} cy={lerp(ny(k + 1, j), ny(k, i), bwdT)} r={5} fill={C.amber} />);
            return out;
          })}
          {cols.map((n, k) => Array.from({length: n}).map((_, i) => (
            <circle key={`n${k}-${i}`} cx={nx(k)} cy={ny(k, i)} r={11} fill={C.bg} stroke={k === 4 ? C.amber : C.cyan} strokeWidth={2}
              style={{filter: bwd > 0 && bwd * 4 > 4 - k - 1 ? `drop-shadow(0 0 6px ${C.amber})` : undefined}} />
          )))}
          <Arrow x1={420} y1={860} x2={1500} y2={860} p={fwd} color={C.cyan} />
          <Arrow x1={1500} y1={895} x2={420} y2={895} p={bwd} color={C.amber} />
        </Svg>
        <Txt x={1520} y={846} w={300} size={20} font="mono" color={C.cyan} o={fwd}>forward: loss</Txt>
        <Txt x={405} y={881} w={260} size={20} font="mono" color={C.amber} align="right" o={bwd}>backward: gradients</Txt>
        <Tag x={1500} y={170} kind="schematic" anchor="right">network drawn small; real one has 545,098 parameters</Tag>
      </Abs>

      {/* Measured loss slice */}
      <Abs x={0} y={0} o={lsO}>
        <Svg>
          <LossSurface view={view} draw={prog(f, s[6], 50)} />
          {pts.length > 1 && <path d={linePath(pts)} stroke={C.cream} strokeWidth={3} fill="none" strokeDasharray="7 6" />}
          {pathP > 0 && <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r={11} fill={C.cream} stroke={C.bg} strokeWidth={2} />}
        </Svg>
        <Eq x={1520} y={320} size={50} o={prog(f, s[6] + 30, 20)}>
          <V>w</V> ← <V>w</V> − <V c={C.amber}>η</V> ∇<V>L</V>(<V>w</V>)
        </Eq>
        <Txt x={1520} y={410} w={520} align="center" size={20} color={C.muted} font="mono" o={prog(f, s[6] + 50, 16)}>
          η = learning rate (0.01 here)
        </Txt>
        <Tag x={1250} y={520} kind="measured" o={prog(f, s[6] + 20, 16)}>loss slice around the baseline</Tag>
        <Txt x={1250} y={556} w={620} size={17} color={C.muted} font="mono" o={prog(f, s[6] + 24, 16)} lh={1.5}>
          25×25 grid · 2 filter-normalized random directions · cross-entropy on 1,000 training images · height = log loss
        </Txt>
        <Tag x={1250} y={680} kind="illustrative" o={prog(f, word(6, 'path') - 6, 14)}>dashed path: descent on this slice, not recorded steps</Tag>
      </Abs>

      {/* Batches and epochs */}
      <Abs x={0} y={0} o={epO}>
        <Txt x={960} y={262} w={1300} align="center" size={34} font="serif">
          50,000 images ÷ <span style={{color: C.cyan}}>64</span> per batch → <span style={{color: C.amber}}>782</span> steps = 1 epoch
        </Txt>
        <Svg>
          {Array.from({length: 782}).map((_, i) => {
            const c = i % 46, r = Math.floor(i / 46);
            const on = i < steps;
            return <rect key={i} x={362 + c * 26} y={360 + r * 26} width={22} height={22} rx={2} fill={on ? (i === steps - 1 ? C.amber : C.cyanDim) : 'none'}
              stroke={C.line} strokeWidth={1} />;
          })}
        </Svg>
        <Txt x={1558} y={812} w={600} align="right" size={22} font="mono" color={C.muted}>step {steps.toLocaleString('en-US')} / 782 · last batch holds 16</Txt>
      </Abs>

      {/* Baseline learning curve */}
      <Abs x={0} y={0} o={blO}>
        <Svg>
          <Axes x={ch.x} y={ch.y} w={ch.w} h={ch.h} xd={[0, 10]} yd={[0, 100]} xTicks={[1, 2, 4, 6, 8, 10]} yTicks={[0, 20, 40, 60, 80, 100]}
            xLabel="epoch" yLabel="accuracy (%)" />
          <line x1={sx(0)} x2={sx(10)} y1={sy(10)} y2={sy(10)} stroke={C.dim} strokeDasharray="6 6" />
          <text x={sx(10) + 10} y={sy(10) + 6} fill={C.dim} fontFamily={F.mono} fontSize={18}>chance 10%</text>
          <DrawPath d={linePath(curve.map((c) => [sx(c.epoch), sy(c.train)]))} p={drawP} stroke={C.muted} width={2} dash="5 6" />
          <DrawPath d={linePath(curve.map((c) => [sx(c.epoch), sy(c.test)]))} p={drawP} stroke={C.amber} width={4} />
          {curve.map((c, i) => (
            <circle key={i} cx={sx(c.epoch)} cy={sy(c.test)} r={5} fill={C.amber} opacity={drawP * 10 > i ? 1 : 0} />
          ))}
          <text x={sx(10) + 14} y={sy(curve[9].train) - 4} fill={C.muted} fontFamily={F.mono} fontSize={18} opacity={drawP}>train {curve[9].train.toFixed(2)}</text>
        </Svg>
        <Txt x={sx(10) + 14} y={sy(curve[9].test) - 2} w={300} size={34} font="serif" color={C.amber} o={prog(f, word(8, 'percent') - 8, 14)}>
          {curve[9].test.toFixed(2)}%
        </Txt>
        <Tag x={ch.x} y={ch.y - 46} kind="test" o={prog(f, s[8] + 20, 14)}>classroom baseline · batch 64 · lr 0.01 · seed 0 · ../runs/baseline.log</Tag>
      </Abs>
    </AbsoluteFill>
  );
};
