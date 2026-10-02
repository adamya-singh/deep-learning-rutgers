import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {lerp, prog, window} from '../lib/anim';
import {Abs, Axes, DrawPath, Panel, Svg, Tag, Txt, linePath, scales} from '../components/ui';
import {R, dep} from '../lib/data';

export const Recipe: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3].map(seg);

  // Beat A: the split
  const splitO = window(f, s[0] - 6, s[1] + 4, 16);
  const X = 310, Wd = 1300, Y = 360;
  const u = Wd / 60000;
  const carve = prog(f, word(0, 'stratified') - 4, 30);
  const testTag = prog(f, word(0, 'reserved') - 6, 16);
  const pristine = prog(f, word(0, 'pristine') - 10, 16);

  // Beat B: LR schedule + recipe card
  const lrO = window(f, s[1] - 2, s[2] + 4, 16);
  const lr = R.depth.lr_schedule;
  const lb = {x: 220, y: 300, w: 820, h: 400};
  const lsc = scales(lb.x, lb.y, lb.w, lb.h, [1, 160], [0, 0.105]);
  const lrDraw = prog(f, word(1, 'warms') - 6, 70);

  // Beat C: three seeds, final-10 window
  const seedO = window(f, s[2] - 2, s[3] + 4, 16);
  const p2 = dep('plain', 2);
  const sb = {x: 300, y: 290, w: 1100, h: 440};
  const ssc = scales(sb.x, sb.y, sb.w, sb.h, [1, 160], [55, 85]);
  const seedDraw = prog(f, s[2] + 6, 60);
  const winO = prog(f, word(2, 'last') - 4, 16);

  // Beat D: two numbers, two metrics
  const cmpO = prog(f, s[3] - 2, 16);

  return (
    <AbsoluteFill>
      <Abs x={0} y={0} o={splitO}>
        <Svg>
          <rect x={X} y={Y} width={45000 * u} height={60} fill={C.cyan} opacity={0.8} rx={3} />
          <rect x={X + 45000 * u + 3 + carve * 0} y={Y + carve * 120} width={5000 * u - 3} height={60} fill={C.cream} opacity={0.9} rx={3} />
          <rect x={X + 50000 * u + 6} y={Y} width={10000 * u - 6} height={60} fill={C.amber} opacity={lerp(0.9, 0.35, testTag)} rx={3} />
          {Array.from({length: 10}).map((_, i) => (
            <rect key={i} x={X + 45000 * u + 3 + i * ((5000 * u - 3) / 10)} y={Y + 190} width={(5000 * u - 3) / 10 - 3} height={46}
              fill={C.cream} opacity={0.6 * prog(f, word(0, 'stratified') + 20 + i * 2, 10)} />
          ))}
        </Svg>
        <Txt x={X} y={Y - 50} w={800} size={24} font="mono" color={C.muted}>official training set: 50,000</Txt>
        <Txt x={X + 50000 * u + 6} y={Y - 50} w={300} size={24} font="mono" color={C.amber}>test: 10,000</Txt>
        <Txt x={X} y={Y + 76} w={700} size={24} font="mono" color={C.cyan} o={carve}>train 45,000</Txt>
        <Txt x={X + 45000 * u - 360} y={Y + 254} w={700} size={22} font="mono" color={C.cream} o={carve} align="left">
          validation 5,000 · 500 per class · split seed 42
        </Txt>
        <Abs x={X + 50000 * u + 20} y={Y + 80} w={470} o={testTag}>
          <div style={{fontFamily: F.mono, fontSize: 20, color: C.amber, lineHeight: 1.5}}>
            reserved for final assessment
            <div style={{color: C.muted, opacity: pristine}}>…but already consulted in the earlier exploratory suite → not a pristine holdout</div>
          </div>
        </Abs>
        <Txt x={960} y={720} w={1400} align="center" size={34} font="serif" italic o={prog(f, word(0, 'decision') - 4, 16)}>
          Every decision from here on is made on <span style={{color: C.cream, fontStyle: 'normal'}}>validation</span>.
        </Txt>
        <Tag x={X} y={810} kind="code" o={prog(f, word(0, 'decision'), 14)}>../depth_cnn.py split_indices · ../runs/depth/split.json</Tag>
      </Abs>

      <Abs x={0} y={0} o={lrO}>
        <Svg>
          <Axes x={lb.x} y={lb.y} w={lb.w} h={lb.h} xd={[1, 160]} yd={[0, 0.105]} xTicks={[1, 5, 40, 80, 120, 160]} yTicks={[0, 0.05, 0.1]}
            xLabel="epoch" yLabel="learning rate" />
          <DrawPath d={linePath(lr.map((v, i) => [lsc.sx(i + 1), lsc.sy(v)]))} p={lrDraw} stroke={C.amber} width={3.5} />
        </Svg>
        <Txt x={lsc.sx(5) + 12} y={lsc.sy(0.1) - 34} w={300} size={18} font="mono" color={C.amber} o={prog(f, word(1, 'warms'), 14)}>warmup 0.01 → 0.1</Txt>
        <Txt x={lsc.sx(160) - 300} y={lsc.sy(0) - 40} w={300} align="right" size={18} font="mono" color={C.amber} o={prog(f, word(1, 'cosine'), 14)}>cosine → 0.0001 at 160</Txt>
        <Tag x={lb.x} y={lb.y - 50} kind="measured">optimizer/lr per epoch, ../runs/depth/plain/conv-2/seed-0/metrics.jsonl</Tag>
        <Panel x={1140} y={300} w={600} h={330} o={prog(f, s[1] + 4, 16)}>
          <div style={{padding: '26px 32px', fontFamily: F.mono, fontSize: 22, color: C.cream, lineHeight: 1.75}}>
            {[
              ['batch', '128 (352 steps/epoch)'],
              ['optimizer', 'SGD, momentum 0.9'],
              ['weight decay', '0.0005'],
              ['schedule', '5-epoch warmup, cosine'],
              ['epochs', '160'],
              ['BatchNorm', 'after every conv'],
              ['augmentation', 'none'],
            ].map(([k, v], i) => (
              <div key={k} style={{display: 'flex', justifyContent: 'space-between', opacity: prog(f, s[1] + 10 + i * 8, 12)}}>
                <span style={{color: C.muted}}>{k}</span><span>{v}</span>
              </div>
            ))}
          </div>
        </Panel>
      </Abs>

      <Abs x={0} y={0} o={seedO}>
        <Svg>
          <Axes x={sb.x} y={sb.y} w={sb.w} h={sb.h} xd={[1, 160]} yd={[55, 85]} xTicks={[1, 40, 80, 120, 160]} yTicks={[55, 65, 75, 85]}
            xLabel="epoch" yLabel="validation accuracy (%)" />
          <rect x={ssc.sx(151)} y={sb.y} width={ssc.sx(160) - ssc.sx(151)} height={sb.h} fill={C.cream} opacity={0.12 * winO} />
          {p2.seed_val_curves.map((c, k) => (
            <DrawPath key={k} d={linePath(c.map((v, i) => [ssc.sx(i + 1), ssc.sy(v)]))} p={seedDraw}
              stroke={[C.cyan, C.blue, C.cream][k]} width={1.8} o={0.85} />
          ))}
          <line x1={ssc.sx(151)} x2={ssc.sx(160)} y1={ssc.sy(p2.mean)} y2={ssc.sy(p2.mean)} stroke={C.amber} strokeWidth={4} opacity={winO} />
        </Svg>
        {['seed 0', 'seed 1', 'seed 2'].map((t, k) => (
          <Txt key={t} x={sb.x + 20 + k * 120} y={sb.y + 10} w={120} size={19} font="mono" color={[C.cyan, C.blue, C.cream][k]} o={seedDraw}>{t}</Txt>
        ))}
        <Txt x={ssc.sx(160) + 24} y={ssc.sy(p2.mean) - 30} w={360} size={20} font="mono" color={C.cream} o={winO} lh={1.4}>
          mean of epochs 151–160,<br />averaged over seeds<br />
          <span style={{fontFamily: F.serif, fontSize: 36, color: C.amber}}>{p2.mean.toFixed(2)} ± {p2.sd.toFixed(2)}</span>
        </Txt>
        <Tag x={sb.x} y={sb.y - 50} kind="validation">2-conv + BN under the new recipe · ± = sample SD across seeds</Tag>
      </Abs>

      <Abs x={0} y={0} o={cmpO * (1 - prog(f, end - 6, 8))}>
        <Panel x={250} y={300} w={640} h={330} border={C.amber}>
          <div style={{padding: 34, fontFamily: F.sans, color: C.cream}}>
            <div style={{fontFamily: F.mono, fontSize: 20, color: C.amber, letterSpacing: 1.5}}>TEST · classroom run</div>
            <div style={{fontFamily: F.serif, fontSize: 96, marginTop: 10}}>{R.classroom.baseline_b64[9].test.toFixed(2)}%</div>
            <div style={{fontSize: 22, color: C.muted, marginTop: 6, lineHeight: 1.5}}>2 conv · no BN · batch 64 · lr 0.01 · 10 epochs · 50k train</div>
          </div>
        </Panel>
        <Panel x={1030} y={300} w={640} h={330} border={C.cyan}>
          <div style={{padding: 34, fontFamily: F.sans, color: C.cream, opacity: prog(f, word(3, 'scores') - 4, 14)}}>
            <div style={{fontFamily: F.mono, fontSize: 20, color: C.cyan, letterSpacing: 1.5}}>VALIDATION · new recipe</div>
            <div style={{fontFamily: F.serif, fontSize: 96, marginTop: 10}}>{p2.mean.toFixed(2)}%</div>
            <div style={{fontSize: 22, color: C.muted, marginTop: 6, lineHeight: 1.5}}>2 conv + BN · batch 128 · SGD+momentum · 160 epochs · 45k train · 3 seeds</div>
          </div>
        </Panel>
        <Txt x={960} y={420} w={140} align="center" size={60} font="serif" color={C.muted}>≠</Txt>
        <Txt x={960} y={690} w={1500} align="center" size={32} font="serif" italic o={prog(f, word(3, 'split') - 4, 16)}>
          Split, metric, and several training knobs changed at once — <span style={{color: C.amber}}>not an isolated ablation</span>.
        </Txt>
      </Abs>
    </AbsoluteFill>
  );
};
