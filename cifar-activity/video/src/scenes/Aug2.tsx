import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {prog, window} from '../lib/anim';
import {Abs, Axes, Clip, DrawPath, Panel, Pix, Svg, Tag, Txt, linePath, scales, signed} from '../components/ui';
import {aug, fu} from '../lib/data';

const CANDS = [
  {id: 'winner-none', label: 'none', color: C.cream, cue: 'No'},
  {id: 'winner-flip', label: 'flip', color: C.cyan, cue: 'Flip'},
  {id: 'winner-crop-flip', label: 'crop + flip', color: C.amber, cue: 'Crop'},
];
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;

export const Aug2: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2].map(seg);

  const introO = window(f, s[0] - 6, s[1] + 4, 16);
  const ch = {x: 300, y: 300, w: 980, h: 430};
  const {sx, sy} = scales(ch.x, ch.y, ch.w, ch.h, [1, 160], [60, 100]);
  const chartO = window(f, s[1] - 4, s[2] + 6, 16);
  const cmpO = prog(f, s[2], 16) * (1 - prog(f, end - 6, 8));
  const smallDelta = mean(aug('crop-flip').delta_vs_none);
  const bigDelta = mean(fu('winner-crop-flip').paired_gain);

  return (
    <AbsoluteFill>
      <Abs x={0} y={0} o={introO}>
        {[0, 1, 4].map((v, i) => <Pix key={v} src={`aug_horse_${v}.png`} x={260 + i * 150} y={380} w={130} border={C.amber} />)}
        <Txt x={260} y={530} w={440} size={22} font="mono" color={C.amber}>crop + flip</Txt>
        <Txt x={260} y={580} w={700} size={22} font="mono" color={C.muted}>
          on the small CNN: {signed(smallDelta)} pp (test, exploratory)
        </Txt>
        <Svg>
          {Array.from({length: 16}).map((_, i) => {
            const x = 1000 + i * 46 + (i >= 8 ? 40 : 0);
            return (
              <g key={i} opacity={prog(f, s[0] + 20 + i * 2, 10)}>
                <rect x={x} y={420} width={30} height={i >= 8 ? 110 : 150} rx={3} fill={i >= 8 ? C.blue : C.cyan} opacity={0.75} />
                <path d={`M${x - 6},${415} Q${x + 15},${380} ${x + 36},${415}`} stroke={C.amber} strokeWidth={1.5} fill="none" opacity={0.7} />
              </g>
            );
          })}
        </Svg>
        <Txt x={1000} y={600} w={800} size={24} font="mono" o={prog(f, s[0] + 30, 14)}>residual 32 · 16 blocks · recipe unchanged</Txt>
        <Txt x={1000} y={640} w={800} size={20} font="mono" color={C.muted} o={prog(f, s[0] + 40, 14)}>
          same seeds and initial weights as the unaugmented runs
        </Txt>
      </Abs>

      <Abs x={0} y={0} o={chartO}>
        <Svg>
          <Axes x={ch.x} y={ch.y} w={ch.w} h={ch.h} xd={[1, 160]} yd={[60, 100]} xTicks={[1, 40, 80, 120, 160]} yTicks={[60, 70, 80, 90, 100]}
            xLabel="epoch" yLabel="validation accuracy (%)" />
          <Clip x={ch.x} y={ch.y} w={ch.w} h={ch.h}>
            {CANDS.map((c) => {
              const cand = fu(c.id);
              const t = word(1, c.cue);
              return <DrawPath key={c.id} d={linePath(cand.val_curve.map((v, i) => [sx(i + 1), sy(v)]))} p={prog(f, t - 30, 40)} stroke={c.color} width={2.5} />;
            })}
          </Clip>
        </Svg>
        {CANDS.map((c, i) => {
          const cand = fu(c.id);
          return (
            <Txt key={c.id} x={ch.x + ch.w + 20} y={sy(cand.mean) - 16 + (i === 0 ? 10 : 0)} w={420} size={22} font="mono" color={c.color}
              o={prog(f, word(1, c.cue) + 6, 12)}>
              {c.label} {cand.mean.toFixed(2)} ± {cand.sd?.toFixed(2)}
            </Txt>
          );
        })}
        {/* paired seed dots */}
        <Svg>
          {[0, 1, 2].map((seed) => {
            const pts = CANDS.map((c, i) => [1670 + i * 75, sy(fu(c.id).per_seed.find((p) => p.seed === seed)!.final10)] as [number, number]);
            return (
              <g key={seed} opacity={prog(f, word(1, 'held') - 4, 16)}>
                <path d={linePath(pts)} stroke={C.dim} strokeWidth={1.5} fill="none" />
                {pts.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={6} fill={CANDS[i].color} />)}
              </g>
            );
          })}
        </Svg>
        <Txt x={1600} y={sy(80)} w={300} size={18} font="mono" color={C.muted} o={prog(f, word(1, 'held'), 14)} align="left">
          lines = paired seeds<br />(final-10 mean)
        </Txt>
        <Tag x={ch.x} y={ch.y - 50} kind="validation">residual 32 · seed-mean curves · ../runs/followup/winner-*/seed-*/metrics.jsonl</Tag>
      </Abs>

      <Abs x={0} y={0} o={cmpO}>
        <Panel x={260} y={300} w={640} h={260} border={C.line}>
          <div style={{padding: 30, fontFamily: F.sans, color: C.cream}}>
            <div style={{fontFamily: F.mono, fontSize: 18, color: C.muted, letterSpacing: 1.3}}>SMALL CNN · TEST · EXPLORATORY</div>
            <div style={{fontFamily: F.serif, fontSize: 80, color: C.amber}}>{signed(smallDelta)} pp</div>
            <div style={{fontSize: 22, color: C.muted}}>crop + flip vs none · batch 512 · 80 epochs</div>
          </div>
        </Panel>
        <Panel x={1020} y={300} w={640} h={260} border={C.cyan}>
          <div style={{padding: 30, fontFamily: F.sans, color: C.cream}}>
            <div style={{fontFamily: F.mono, fontSize: 18, color: C.cyan, letterSpacing: 1.3}}>RESIDUAL 32 · VALIDATION</div>
            <div style={{fontFamily: F.serif, fontSize: 80, color: C.cyan}}>{signed(bigDelta)} pp</div>
            <div style={{fontSize: 22, color: C.muted}}>crop + flip vs none · stricter recipe · 160 epochs</div>
          </div>
        </Panel>
        <Txt x={960} y={620} w={1500} align="center" size={32} font="serif" italic o={prog(f, word(2, 'Plausibly') - 4, 16)}>
          Plausibly: more capacity + a stronger recipe can use the variety.
        </Txt>
        <Txt x={960} y={690} w={1500} align="center" size={24} font="mono" color={C.muted} o={prog(f, word(2, 'prove') - 4, 16)}>
          not proven · model, recipe, and split all changed together
        </Txt>
      </Abs>
    </AbsoluteFill>
  );
};
