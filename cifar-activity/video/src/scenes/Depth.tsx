import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {lerp, prog, window} from '../lib/anim';
import {Abs, Axes, Clip, DrawPath, Panel, Pix, Svg, Tag, Txt, linePath, scales} from '../components/ui';
import {DepthChart} from '../components/DepthChart';
import {DEPTHS, dep} from '../lib/data';

/** Two-stage plain network with n convolutions (n/2 per stage, or 1 per stage at n = 2). */
const Stack: React.FC<{n: number; x: number; y: number; o?: number}> = ({n, x, y, o = 1}) => {
  const per = Math.max(1, n / 2);
  const stageW = 560;
  const gap = Math.min(30, stageW / per);
  const bw = Math.max(6, Math.min(18, gap - 6));
  return (
    <div style={{position: 'absolute', left: x, top: y, opacity: o}}>
      {[0, 1].map((st) => (
        <div key={st} style={{position: 'absolute', left: st * (stageW + 120), top: 0}}>
          <div style={{position: 'absolute', left: 0, top: -48, fontFamily: F.mono, fontSize: 20, color: st ? C.blue : C.cyan, whiteSpace: 'nowrap'}}>
            stage {st + 1} · {st ? 64 : 32} channels
          </div>
          {Array.from({length: per}).map((_, i) => (
            <div key={i} style={{position: 'absolute', left: i * gap, top: st ? 20 : 0, width: bw, height: st ? 140 : 180,
              background: st ? C.blue : C.cyan, opacity: 0.75, borderRadius: 2}} />
          ))}
          <div style={{position: 'absolute', left: per * gap + 14, top: 50, fontFamily: F.mono, fontSize: 16, color: C.muted}}>pool</div>
        </div>
      ))}
      <div style={{position: 'absolute', left: 2 * (stageW + 120) - 20, top: 60, fontFamily: F.mono, fontSize: 20, color: C.muted, whiteSpace: 'nowrap'}}>
        → 4096 → 128 → 10
      </div>
    </div>
  );
};

export const Depth: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3, 4].map(seg);

  const stackO = window(f, s[0] - 6, s[1] + 4, 16);
  const step = Math.max(0, Math.min(4, Math.floor((f - s[0] - 40) / 26)));
  const n = DEPTHS[step];

  const rfO = window(f, s[1] - 2, s[2] + 4, 16);
  const P = 12.5, FX = 300, FY = 330;
  const rf = [prog(f, s[1] + 10, 14), prog(f, word(1, '5') - 4, 14), prog(f, word(1, '7') - 4, 14)];

  const chartO = window(f, s[2] - 4, end + 20, 16);
  const shrink = prog(f, s[3], 30);
  const reveal = [
    prog(f, word(2, 'Two'), 14), prog(f, word(2, 'Four'), 14), prog(f, word(2, 'Eight'), 14),
    prog(f, word(2, 'Sixteen'), 14), prog(f, word(2, 'drops'), 18),
  ];
  const cx = lerp(380, 170, shrink), cw = lerp(1160, 760, shrink), cy = 290, chh = lerp(470, 430, shrink);

  const p16 = dep('plain', 16), p32 = dep('plain', 32);
  const cb = {x: 1100, y: 300, w: 660, h: 380};
  const csc = scales(cb.x, cb.y, cb.w, cb.h, [1, 160], [40, 100]);
  const curveO = prog(f, s[3] + 20, 16) * (1 - prog(f, s[4] - 4, 14));
  const cmpO = prog(f, s[4], 16);

  return (
    <AbsoluteFill>
      <Abs x={0} y={0} o={stackO}>
        <Stack n={n} x={260} y={420} />
        <Txt x={960} y={720} w={800} align="center" size={44} font="serif">
          <span style={{color: C.amber}}>{n}</span> convolutions
        </Txt>
        <Txt x={960} y={790} w={1200} align="center" size={20} font="mono" color={C.muted}>
          plain sweep 2 → 4 → 8 → 16 → 32 · BatchNorm after every conv · ../depth_cnn.py
        </Txt>
      </Abs>

      <Abs x={0} y={0} o={rfO}>
        <Pix src="feat_frog.png" x={FX} y={FY} w={400} o={0.85} />
        <Svg>
          {[3, 5, 7].map((k, i) => (
            <rect key={k} x={FX + (16 - (k - 1) / 2) * P} y={FY + (16 - (k - 1) / 2) * P} width={k * P} height={k * P} fill="none"
              stroke={[C.cyan, C.blue, C.amber][i]} strokeWidth={3} opacity={rf[i]} />
          ))}
          {/* cone through stacked layers (schematic) */}
          {[0, 1, 2, 3].map((L) => {
            const yL = 760 - L * 130;
            const half = 3 - L;
            return (
              <g key={L} opacity={L === 0 ? 1 : rf[Math.min(2, L - 1)]}>
                {Array.from({length: 9}).map((_, i) => (
                  <circle key={i} cx={1050 + i * 70} cy={yL} r={10} fill={Math.abs(i - 4) <= half ? (L === 3 ? C.amber : C.cyanDim) : C.bg}
                    stroke={C.line} strokeWidth={2} />
                ))}
                {L > 0 && [-1, 0, 1].flatMap((dx) =>
                  Array.from({length: 2 * half + 1}).map((_, j) => {
                    const i = 4 - half + j;
                    const k = i + dx;
                    return <line key={`${i}-${dx}`} x1={1050 + i * 70} y1={yL + 10} x2={1050 + k * 70} y2={yL + 120} stroke={C.cyanDim} strokeWidth={1.2} opacity={0.7} />;
                  }),
                )}
                <text x={1000} y={yL + 6} fill={C.muted} fontFamily={F.mono} fontSize={17} textAnchor="end">{L === 0 ? 'input' : `layer ${L}`}</text>
              </g>
            );
          })}
        </Svg>
        <Txt x={FX} y={FY + 420} w={600} size={22} font="mono" color={C.muted}>
          one unit sees <span style={{color: C.cyan}}>3×3</span> → <span style={{color: C.blue}}>5×5</span> → <span style={{color: C.amber}}>7×7</span> input pixels
        </Txt>
        <Tag x={1050} y={290} kind="schematic">receptive field of one unit through stacked 3×3 layers</Tag>
      </Abs>

      <Abs x={0} y={0} o={chartO}>
        <Svg>
          <DepthChart x={cx} y={cy} w={cw} h={chh} plain={reveal} />
        </Svg>
        <Tag x={cx} y={cy - 50} kind="validation">plain networks · final-10-epoch mean · 3 seeds · ± SD</Tag>
        <Txt x={cx + cw - 360} y={cy + chh - 120} w={360} align="right" size={30} font="serif" color={C.amber} o={reveal[4] * (1 - shrink)}>
          −4.10 pp, all 3 seeds decline
        </Txt>
      </Abs>

      <Abs x={0} y={0} o={curveO}>
        <Svg>
          <Axes x={cb.x} y={cb.y} w={cb.w} h={cb.h} xd={[1, 160]} yd={[40, 100]} xTicks={[1, 80, 160]} yTicks={[40, 60, 80, 100]} xLabel="epoch" />
          <Clip x={cb.x} y={cb.y} w={cb.w} h={cb.h}>
          {[[p16, C.cream], [p32, C.amber]].map(([c, col], k) => {
            const cand = c as typeof p16;
            return (
              <g key={k}>
                <DrawPath d={linePath(cand.train_curve.map((v, i) => [csc.sx(i + 1), csc.sy(v)]))} p={prog(f, s[3] + 24, 50)} stroke={col as string} width={2} dash="5 5" />
                <DrawPath d={linePath(cand.val_curve.map((v, i) => [csc.sx(i + 1), csc.sy(v)]))} p={prog(f, s[3] + 24, 50)} stroke={col as string} width={3} />
              </g>
            );
          })}
          </Clip>
        </Svg>
        <Txt x={cb.x} y={cb.y - 50} w={760} size={20} font="mono" color={C.muted}>
          <span style={{color: C.cream}}>plain 16</span> vs <span style={{color: C.amber}}>plain 32</span> · dashed train · solid validation
        </Txt>
        <Txt x={cb.x} y={cb.y + cb.h + 64} w={700} size={20} font="mono" color={C.muted} o={prog(f, s[3] + 40, 14)}>
          final clean train accuracy: {DEPTHS.every((d) => dep('plain', d).final_train >= 99.95) ? '100.0% at all five depths' : DEPTHS.map((d) => dep('plain', d).final_train.toFixed(1)).join(' · ')}
        </Txt>
        <Txt x={cb.x} y={cb.y + cb.h + 100} w={760} size={28} font="serif" italic o={prog(f, word(3, 'fitting'), 14)}>
          Fitting the training set isn't generalizing.
        </Txt>
      </Abs>

      <Abs x={0} y={0} o={cmpO * (1 - prog(f, end - 6, 8))}>
        <Panel x={1000} y={290} w={800} h={470} border={C.line}>
          <div style={{padding: '28px 34px', fontFamily: F.sans, color: C.cream, fontSize: 24, lineHeight: 1.45}}>
            <div style={{fontFamily: F.mono, fontSize: 18, color: C.muted, letterSpacing: 1.2}}>ORIGINAL RESNET PAPER (He et al., 2015)</div>
            <div style={{marginTop: 8}}>Deeper plain nets had <span style={{color: C.amber}}>higher training error</span>: an optimization problem.</div>
            <div style={{fontFamily: F.mono, fontSize: 18, color: C.muted, letterSpacing: 1.2, marginTop: 28, opacity: prog(f, word(4, 'ours'), 14)}}>OUR SWEEP</div>
            <div style={{marginTop: 8, opacity: prog(f, word(4, 'ours'), 14)}}>Training ≈ 100% at every depth. The decline shows up in <span style={{color: C.cyan}}>validation</span>. Cause not diagnosed; no gradient measurements.</div>
            <div style={{marginTop: 28, fontFamily: F.serif, fontStyle: 'italic', fontSize: 28, opacity: prog(f, word(4, 'Residuals'), 14)}}>
              Residual connections: a plausible fix to <span style={{color: C.amber, fontStyle: 'normal'}}>test</span>, not a diagnosis.
            </div>
          </div>
        </Panel>
        <Txt x={1000} y={780} w={800} size={17} font="mono" color={C.dim}>arxiv.org/abs/1512.03385</Txt>
      </Abs>
    </AbsoluteFill>
  );
};
