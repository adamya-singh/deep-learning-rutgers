import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {linear, prog, window} from '../lib/anim';
import {Abs, Axes, DrawPath, Svg, Tag, Txt, linePath, scales} from '../components/ui';
import {R} from '../lib/data';

const B64 = R.classroom.baseline_b64;
const T512 = R.classroom.tuned_b512_10ep;
const M512 = R.classroom.b512_80ep;

export const Batch: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3].map(seg);

  const laneO = window(f, s[0] - 6, s[2] + 4, 16);
  const run = linear(f, s[0] + 20, 150);
  const chartO = prog(f, s[2] - 4, 16) * (1 - 0.6 * prog(f, s[3], 20));
  const passO = prog(f, s[3] + 4, 16);

  const lanes = [
    {y: 300, label: 'batch 64', block: 10, secs: 22.39, color: C.cyan, updates: 7820, acc: B64[9].test},
    {y: 470, label: 'batch 512', block: 80, secs: 5.21, color: C.amber, updates: 980, acc: T512[9].test},
  ];
  const ch = {x: 360, y: 270, w: 1100, h: 430};
  const {sx, sy} = scales(ch.x, ch.y, ch.w, ch.h, [0, 8000], [20, 75]);
  const draw = prog(f, s[2] + 6, 60);

  return (
    <AbsoluteFill>
      {/* Two lanes: images per step vs. number of steps */}
      <Abs x={0} y={0} o={laneO}>
        {lanes.map((l, k) => {
          const t = Math.min(1, (run * 24) / l.secs); // shared simulated clock; each lane stops at its measured time
          const shownUpdates = Math.round(Math.min(1, t) * l.updates);
          return (
            <React.Fragment key={l.label}>
              <Txt x={150} y={l.y + 6} w={200} size={26} font="mono" color={l.color}>{l.label}</Txt>
              <Svg>
                <rect x={360} y={l.y} width={1100} height={58} fill="none" stroke={C.line} />
                {Array.from({length: 40}).map((_, i) => {
                  const x = 360 + ((i * (l.block + 6) + run * 2600 * (l.secs < 10 ? 1.9 : 0.42)) % 1100);
                  return x + l.block < 1460 ? <rect key={i} x={x} y={l.y + 10} width={l.block} height={38} rx={2} fill={l.color} opacity={0.5} /> : null;
                })}
              </Svg>
              <Txt x={1490} y={l.y + 4} w={300} size={30} font="mono">{(t * l.secs).toFixed(2)} s</Txt>
              <Txt x={360} y={l.y + 70} w={1100} size={20} font="mono" color={C.muted}
                o={prog(f, s[1] + 4, 14)}>
                {shownUpdates.toLocaleString('en-US')} optimizer updates in 10 epochs → test accuracy{' '}
                <span style={{color: l.color}}>{l.acc.toFixed(2)}%</span>
              </Txt>
              {/* one dot per 20 updates */}
              <Svg o={prog(f, s[1] + 10, 14)}>
                {Array.from({length: Math.round(l.updates / 20)}).map((_, i) => (
                  <circle key={i} cx={362 + (i % 196) * 5.6} cy={l.y + 116 + Math.floor(i / 196) * 6} r={1.8} fill={l.color}
                    opacity={i < (shownUpdates / 20) * prog(f, s[1] + 10, 50) ? 0.95 : 0} />
                ))}
              </Svg>
              {k === 1 && (
                <Txt x={1490} y={l.y + 104} w={360} size={18} font="mono" color={C.muted} o={prog(f, s[1] + 30, 14)}>1 dot = 20 updates</Txt>
              )}
            </React.Fragment>
          );
        })}
        <Tag x={360} y={720} kind="measured" o={prog(f, s[0] + 30, 14)}>train+eval time, RTX 3090 · ../BENCHMARKS.md</Tag>
        <Txt x={960} y={780} w={1400} align="center" size={32} font="serif" italic o={prog(f, word(1, 'Turns') - 4, 16)}>
          Same learning rate, <span style={{color: C.amber}}>8× fewer steps</span>.
        </Txt>
      </Abs>

      {/* Real test curves against optimizer updates */}
      <Abs x={0} y={0} o={chartO}>
        <Svg>
          <Axes x={ch.x} y={ch.y} w={ch.w} h={ch.h} xd={[0, 8000]} yd={[20, 75]} xTicks={[0, 2000, 4000, 6000, 8000]} yTicks={[20, 30, 40, 50, 60, 70]}
            xLabel="optimizer updates" yLabel="test accuracy (%)" xFmt={(v) => v.toLocaleString('en-US')} />
          <DrawPath d={linePath(M512.map((c) => [sx(c.epoch * 98), sy(c.test)]))} p={draw} stroke={C.amber} width={3} />
          <DrawPath d={linePath(T512.map((c) => [sx(c.epoch * 98), sy(c.test)]))} p={draw} stroke={C.brown} width={3} dash="6 6" />
          <DrawPath d={linePath(B64.map((c) => [sx(c.epoch * 782), sy(c.test)]))} p={draw} stroke={C.cyan} width={4} />
          <circle cx={sx(7820)} cy={sy(B64[9].test)} r={7} fill={C.cyan} opacity={draw} />
          <circle cx={sx(7840)} cy={sy(M512[79].test)} r={7} fill={C.amber} opacity={draw} />
          <circle cx={sx(980)} cy={sy(T512[9].test)} r={7} fill={C.brown} opacity={draw} />
          <line x1={sx(980)} y1={sy(T512[9].test) + 8} x2={612} y2={598} stroke={C.brown} strokeWidth={1} opacity={draw * 0.7} />
        </Svg>
        <Txt x={620} y={590} w={700} size={20} font="mono" color={C.brown} o={draw}>
          batch 512 · 10 epochs · 980 updates · {T512[9].test.toFixed(2)}%
        </Txt>
        <Txt x={sx(7820)} y={sy(54)} w={420} align="right" size={20} font="mono" color={C.cyan} o={draw}>
          batch 64 · 10 epochs<br />7,820 updates · {B64[9].test.toFixed(2)}%
        </Txt>
        <Txt x={sx(7840) + 16} y={sy(M512[79].test) - 18} w={300} size={20} font="mono" color={C.amber} o={prog(f, word(2, 'reached'), 14)}>
          batch 512 · 80 epochs<br />7,840 updates<br /><span style={{fontSize: 30, fontFamily: F.serif}}>{M512[79].test.toFixed(2)}%</span>
        </Txt>
        <Tag x={ch.x} y={ch.y - 46} kind="test" o={1}>seed 0 · lr 0.01 · per-epoch logs ../runs/{'{baseline,tuned-b512,step-matched-b512}'}.log</Tag>
      </Abs>

      {/* Image passes: the cost that hides behind 'faster' */}
      <Abs x={0} y={0} o={passO}>
        <Svg>
          <rect x={360} y={800} width={1100 * (0.5 / 4) * prog(f, s[3] + 6, 20)} height={30} fill={C.cyan} opacity={0.8} />
          <rect x={360} y={846} width={1100 * prog(f, s[3] + 14, 30)} height={30} fill={C.amber} opacity={0.8} />
        </Svg>
      </Abs>
      <Abs x={0} y={0} o={passO * (1 - prog(f, end - 8, 8))}>
        <Txt x={150} y={802} w={200} size={18} font="mono" color={C.cyan}>b64 · 10 ep</Txt>
        <Txt x={150} y={848} w={200} size={18} font="mono" color={C.amber}>b512 · 80 ep</Txt>
        <Txt x={1480} y={802} w={420} size={18} font="mono" color={C.muted}>0.5 M image passes · 22.39 s</Txt>
        <Txt x={1480} y={848} w={420} size={18} font="mono" color={C.muted}>4.0 M image passes · 36.85 s</Txt>
      </Abs>
    </AbsoluteFill>
  );
};
