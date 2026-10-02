import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {clamp, lerp, prog, window} from '../lib/anim';
import {Abs, Axes, Clip, DrawPath, Panel, Slab, Svg, Tag, Txt, linePath, scales} from '../components/ui';
import {fu} from '../lib/data';

const BASE = 640;
type El = {x: number; size: number; depth: number; ch: number; res: number; kind: 'input' | 'stem' | 'block'; stage?: number; proj?: boolean; idx?: number};
const SZ: Record<number, number> = {32: 128, 16: 64, 8: 32, 4: 16};
const DP: Record<number, number> = {3: 6, 64: 24, 128: 38, 256: 52, 512: 66};
const EL: El[] = [
  {x: 60, size: 128, depth: 6, ch: 3, res: 32, kind: 'input'},
  {x: 250, size: 128, depth: 24, ch: 64, res: 32, kind: 'stem'},
  ...[0, 1, 2, 3, 4, 5, 6, 7].map((i): El => {
    const stage = Math.floor(i / 2);
    const ch = [64, 128, 256, 512][stage];
    const res = [32, 16, 8, 4][stage];
    const xs = [440, 615, 810, 950, 1105, 1215, 1345, 1445];
    return {x: xs[i], size: SZ[res], depth: DP[ch], ch, res, kind: 'block', stage, proj: i % 2 === 0 && stage > 0, idx: i};
  }),
];
const top = (e: El) => BASE - e.size - e.depth * 0.6;

export const ResNet: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3, 4, 5, 6].map(seg);

  const codeO = window(f, s[0] - 6, s[1] + 10, 16);
  const pipeO = window(f, s[1] - 2, s[5] + 4, 16);
  const stemP = prog(f, s[1] + 6, 20);
  const ghostO = window(f, word(1, 'ImageNet') - 6, s[2] + 10, 16);
  const strike = prog(f, word(1, 'swapped') - 4, 20);
  const stageT = [word(2, '64'), word(2, '128'), word(2, '256'), word(2, '512')];
  const resT = [word(2, '32'), word(2, '16'), word(2, '8'), word(2, '4')];
  const projP = prog(f, word(2, 'learned') - 6, 20);
  const gap = prog(f, s[3] + 10, 40);
  const fc = prog(f, word(3, 'linear') - 4, 20);
  // Count convs in sync with the narration: stem → 16 block convs → classifier.
  const tStem = word(4, 'stem') - 4, t16 = word(4, '16') - 4, tCls = word(4, 'classifier') - 4;
  const countT = tStem;
  const counted = f < tStem ? -1 : f < t16 ? 0 : f < tCls ? clamp(1 + (f - t16) / 2, 1, 16) : 17;
  const tableO = window(f, s[5] - 2, s[6] + 4, 16);
  const resultO = prog(f, s[6] - 4, 16);

  const rn = fu('resnet18-crop-flip');
  const w32 = fu('winner-crop-flip');
  const ch = {x: 300, y: 290, w: 1000, h: 430};
  const {sx, sy} = scales(ch.x, ch.y, ch.w, ch.h, [1, 200], [60, 100]);
  const seed0 = rn.per_seed.find((p) => p.seed === 0)!;
  const n = rn.seeds.length;

  // which conv is highlighted during the count (0 = stem, 1..16 = block convs, 17 = classifier)
  const hi = Math.floor(counted);

  return (
    <AbsoluteFill>
      <Abs x={0} y={0} o={codeO}>
        <div style={{position: 'absolute', left: 330, top: 340, width: 1260, padding: '30px 38px', background: 'rgba(5,10,24,0.85)',
          border: `1px solid ${C.line}`, borderRadius: 10, fontFamily: F.mono, fontSize: 25, color: C.cream, lineHeight: 1.8}}>
          <div>model = resnet18(<span style={{color: C.amber}}>weights=None</span>, num_classes=10)</div>
          <div style={{opacity: prog(f, s[0] + 30, 14)}}>model.conv1 = nn.Conv2d(3, 64, <span style={{color: C.cyan}}>3</span>, stride=<span style={{color: C.cyan}}>1</span>, padding=1, bias=False)</div>
          <div style={{opacity: prog(f, s[0] + 50, 14)}}>model.maxpool = <span style={{color: C.cyan}}>nn.Identity()</span></div>
        </div>
        <Tag x={330} y={590} kind="code">../followup_cnn.py build_model · torchvision 0.22.1 · random initialization, no pretrained weights</Tag>
      </Abs>

      <Abs x={0} y={0} o={pipeO}>
        <Txt x={80} y={104} w={1200} size={19} font="mono" color={C.muted}>CIFAR ResNet-18 · weights=None (random init) · torchvision resnet18 with a CIFAR stem</Txt>
        {EL.map((e, i) => {
          let o = 0;
          if (e.kind === 'input') o = prog(f, s[1], 16);
          else if (e.kind === 'stem') o = stemP;
          else o = prog(f, stageT[e.stage!] - 6 + (e.idx! % 2) * 6, 14);
          const convHi = e.kind === 'stem' ? hi === 0 && f > countT : e.kind === 'block' ? hi >= 1 + e.idx! * 2 && hi <= 2 + e.idx! * 2 && f > countT : false;
          const img = e.kind === 'input' ? 'feat_frog.png' : undefined;
          const col = e.kind === 'input' ? C.cream : e.kind === 'stem' ? C.cyan : [C.cyan, C.blue, C.blue, C.amber][e.stage!];
          return (
            <React.Fragment key={i}>
              <Slab x={e.x} y={BASE - e.size} size={e.size} depth={e.depth} layers={e.kind === 'input' ? 3 : 7} color={col} img={img} o={o}
                glow={convHi ? 0.8 : 0} />
            </React.Fragment>
          );
        })}
        {/* stage labels */}
        {[0, 1, 2, 3].map((st) => {
          const blocks = EL.filter((e) => e.stage === st);
          const x0 = blocks[0].x, x1 = blocks[1].x + blocks[1].size + blocks[1].depth;
          return (
            <React.Fragment key={st}>
              <Txt x={(x0 + x1) / 2} y={360} w={240} align="center" size={20} font="mono" color={C.cream} o={prog(f, stageT[st], 12)}>
                {[64, 128, 256, 512][st]} ch
              </Txt>
              <Txt x={(x0 + x1) / 2} y={BASE + 18} w={240} align="center" size={22} font="mono" color={C.amber} o={prog(f, resT[st] - 2, 12)}>
                {[32, 16, 8, 4][st]}×{[32, 16, 8, 4][st]}
              </Txt>
            </React.Fragment>
          );
        })}
        <Txt x={124} y={BASE + 18} w={200} align="center" size={18} font="mono" color={C.muted} o={prog(f, s[1], 14)}>input 3×32×32</Txt>
        <Txt x={326} y={BASE + 18} w={240} align="center" size={18} font="mono" color={C.cyan} o={stemP}>stem 64×32×32</Txt>
        <Txt x={326} y={BASE + 46} w={260} align="center" size={16} font="mono" color={C.muted} o={stemP}>3×3 s1 · BN · ReLU</Txt>
        <Svg>
          {/* shortcuts over every block */}
          {EL.filter((e) => e.kind === 'block').map((e) => {
            const o = prog(f, stageT[e.stage!] + 4 + (e.idx! % 2) * 6, 14);
            const y = top(e) - 22;
            const xa = e.x - 22, xb = e.x + e.size + e.depth + 12;
            const isProj = e.proj;
            return (
              <g key={e.idx} opacity={o}>
                <path d={`M${xa},${top(e) + 10} Q${xa},${y} ${(xa + xb) / 2},${y} Q${xb},${y} ${xb},${top(e) + 10}`} fill="none"
                  stroke={isProj ? C.amber : C.cream} strokeWidth={isProj ? lerp(1.5, 3, projP) : 1.5} strokeDasharray={isProj ? '6 5' : undefined}
                  opacity={isProj ? lerp(0.5, 1, projP) : 0.55} />
                {isProj && (
                  <text x={(xa + xb) / 2} y={y - 12} fill={C.amber} fontFamily={F.mono} fontSize={15} textAnchor="middle" opacity={projP}>1×1 s2 + BN</text>
                )}
              </g>
            );
          })}
          {/* GAP + classifier */}
          {(() => {
            const last = EL[EL.length - 1];
            const sx0 = last.x + last.size + last.depth + 30;
            return (
              <g opacity={gap}>
                <line x1={sx0} y1={BASE - 60} x2={1600} y2={BASE - 60} stroke={C.dim} strokeWidth={2} />
                {Array.from({length: 16}).map((_, i) => (
                  <circle key={i} cx={1620} cy={lerp(BASE - 60, 420 + i * 18, gap)} r={5} fill={C.amber} opacity={0.85} />
                ))}
                <g opacity={fc}>
                  {Array.from({length: 10}).map((_, j) => (
                    <g key={j}>
                      {Array.from({length: 16}).map((__, i) => (
                        <line key={i} x1={1620} y1={420 + i * 18} x2={1760} y2={440 + j * 26} stroke={C.line} strokeWidth={0.7} />
                      ))}
                      <circle cx={1760} cy={440 + j * 26} r={7} fill={C.bg} stroke={C.cream} strokeWidth={2}
                        style={{filter: hi >= 17 && f > countT ? `drop-shadow(0 0 6px ${C.amber})` : undefined}} />
                    </g>
                  ))}
                </g>
              </g>
            );
          })()}
        </Svg>
        <Txt x={1620} y={340} w={260} align="center" size={18} font="mono" color={C.amber} o={gap}>global avg pool</Txt>
        <Txt x={1620} y={366} w={260} align="center" size={16} font="mono" color={C.muted} o={gap}>512×4×4 → 512</Txt>
        <Txt x={1760} y={712} w={220} align="center" size={18} font="mono" color={C.cream} o={fc}>linear 512 → 10</Txt>

        {/* ImageNet stem ghost */}
        <Abs x={0} y={0} o={ghostO}>
          <Panel x={240} y={750} w={1440} h={110} border={C.line}>
            <div style={{padding: '16px 26px', fontFamily: F.mono, fontSize: 21, lineHeight: 1.6}}>
              <div style={{color: C.muted, position: 'relative', display: 'inline-block'}}>
                ImageNet stem: 7×7 conv, stride 2 (32→16) + 3×3 max pool, stride 2 (16→8)
                <div style={{position: 'absolute', left: 0, top: '52%', height: 2.5, width: `${strike * 100}%`, background: C.amber}} />
              </div>
              <div style={{color: C.cyan, opacity: strike}}>CIFAR stem: 3×3 conv, stride 1, no max pool → stays 32×32</div>
            </div>
          </Panel>
        </Abs>
        <Txt x={960} y={770} w={1400} align="center" size={26} font="serif" italic o={window(f, s[3] + 20, s[4], 14)}>
          No flatten, no 128-unit hidden layer: one number per channel, then one linear layer.
        </Txt>
        <Abs x={0} y={0} o={window(f, s[4], s[5] + 4, 14)}>
          <Txt x={960} y={756} w={1400} align="center" size={48} font="serif">
            <span style={{color: C.cyan}}>1</span> stem + <span style={{color: C.blue}}>{Math.max(0, Math.min(16, hi))}</span> block convs
            {hi >= 17 ? <> + <span style={{color: C.amber}}>1</span> classifier = <span style={{color: C.amber}}>18</span></> : <> = {1 + Math.max(0, Math.min(16, hi))}</>}
          </Txt>
          <Txt x={960} y={830} w={1400} align="center" size={20} font="mono" color={C.muted} o={prog(f, word(4, 'convention') - 4, 14)}>
            the 3 dashed 1×1 projection convs are not counted, by convention
          </Txt>
        </Abs>
      </Abs>

      {/* Comparison table */}
      <Abs x={0} y={0} o={tableO}>
        <Panel x={200} y={190} w={1520} h={560} border={C.line}>
          <div style={{padding: '24px 36px', fontFamily: F.sans, fontSize: 23, color: C.cream}}>
            <div style={{display: 'grid', gridTemplateColumns: '300px 1fr 1fr', rowGap: 15, columnGap: 20}}>
              <div />
              <div style={{fontFamily: F.mono, color: C.cream, fontSize: 21}}>custom residual 32</div>
              <div style={{fontFamily: F.mono, color: C.cyan, fontSize: 21}}>CIFAR ResNet-18</div>
              {[
                ['main-path convs', '32', '17 (+3 projection shortcuts)', 'layer'],
                ['channels', '32 → 64', '64 → 128 → 256 → 512', 'longer'],
                ['spatial path', '32 → 16 → 8 (max pool)', '32 → 16 → 8 → 4 (stride 2)', 'longer'],
                ['shape-change shortcut', 'zero-pad channels', 'learned 1×1 conv + BN', 'shortcuts'],
                ['head', 'flatten 4096 → 128 → 10', 'global avg pool 512 → 10', 'global'],
                ['parameters', '1,239,274', '11,173,962 (≈9.0×)', 'parameters'],
                ['epochs', '160', '200', '200'],
                ['augmentation', 'crop + flip', 'crop + flip', '200'],
              ].map(([k, a, b, cue]) => {
                const o = prog(f, word(5, cue) - 6, 12);
                return (
                  <React.Fragment key={k}>
                    <div style={{color: C.muted, fontFamily: F.mono, fontSize: 19, opacity: o}}>{k}</div>
                    <div style={{opacity: o}}>{a}</div>
                    <div style={{opacity: o}}>{b}</div>
                  </React.Fragment>
                );
              })}
            </div>
            <div style={{marginTop: 30, fontFamily: F.serif, fontStyle: 'italic', fontSize: 30, color: C.amber, opacity: prog(f, word(5, 'recipe') - 4, 14)}}>
              A bundled recipe comparison, not an isolated architecture ablation.
            </div>
          </div>
        </Panel>
      </Abs>

      {/* Result */}
      <Abs x={0} y={0} o={resultO}>
        <Svg>
          <Axes x={ch.x} y={ch.y} w={ch.w} h={ch.h} xd={[1, 200]} yd={[60, 100]} xTicks={[1, 40, 80, 120, 160, 200]} yTicks={[60, 70, 80, 90, 100]}
            xLabel="epoch" yLabel="validation accuracy (%)" />
          <rect x={sx(191)} y={ch.y} width={sx(200) - sx(191)} height={ch.h} fill={C.cream} opacity={0.12 * prog(f, s[6] + 50, 14)} />
          <Clip x={ch.x} y={ch.y} w={ch.w} h={ch.h}>
            <DrawPath d={linePath(w32.val_curve.map((v, i) => [sx(i + 1), sy(v)]))} p={prog(f, s[6] + 4, 50)} stroke={C.amber} width={2.5} />
            <DrawPath d={linePath(rn.val_curve.map((v, i) => [sx(i + 1), sy(v)]))} p={prog(f, s[6] + 10, 60)} stroke={C.cyan} width={3} />
          </Clip>
        </Svg>
        <Txt x={sx(160)} y={sy(76)} w={460} align="right" size={19} font="mono" color={C.amber} o={prog(f, s[6] + 40, 14)}>
          residual 32 + crop/flip · {w32.mean.toFixed(2)} (160 ep)
        </Txt>
        <Txt x={ch.x + ch.w + 26} y={sy(rn.mean) - 64} w={520} size={20} font="mono" color={C.cyan} o={prog(f, s[6] + 50, 14)} lh={1.45}>
          ResNet-18 + crop/flip · {n === 1 ? 'seed 0' : `${n} of 3 seeds`}<br />
          final-10 mean{n > 1 ? ', seed-averaged' : ''}<br />
          <span style={{fontFamily: F.serif, fontSize: 46, color: C.cream}}>{rn.mean.toFixed(2)}%</span>
          {n > 1 && rn.sd !== null && <span style={{fontSize: 24, color: C.muted}}> ± {rn.sd.toFixed(2)}</span>}
        </Txt>
        <Txt x={ch.x + ch.w + 26} y={sy(rn.mean) + 70} w={560} size={17} font="mono" color={C.muted} o={prog(f, s[6] + 70, 14)}>
          seed 0: best single epoch {seed0.peak.toFixed(2)} · last epoch {seed0.final_epoch.toFixed(2)} (not used for scoring)
        </Txt>
        <Tag x={ch.x} y={ch.y - 50} kind="validation">{n}/3 seeds complete at snapshot · ../runs/followup/resnet18-crop-flip/seed-*/metrics.jsonl</Tag>
      </Abs>
    </AbsoluteFill>
  );
};
