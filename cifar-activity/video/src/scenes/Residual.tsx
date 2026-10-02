import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {lerp, linear, prog, rand, window} from '../lib/anim';
import {Abs, Arrow, DrawPath, Eq, Panel, Slab, Svg, Tag, Txt, V} from '../components/ui';
import {DepthChart} from '../components/DepthChart';
import {R, dep} from '../lib/data';

const BOXES = [
  {x: 330, t: 'conv 3×3'}, {x: 490, t: 'BN'}, {x: 650, t: 'ReLU'}, {x: 810, t: 'conv 3×3'}, {x: 970, t: 'BN'},
];
const BW = 130, BH = 64;

const BlockRow: React.FC<{y: number; residual: boolean; shortcut: number; fHi: number; o?: number; label: string}> = ({y, residual, shortcut, fHi, o = 1, label}) => (
  <g opacity={o}>
    <text x={120} y={y + 8} fill={C.muted} fontFamily={F.mono} fontSize={20}>{label}</text>
    <circle cx={260} cy={y} r={20} fill={C.bg} stroke={C.cream} strokeWidth={2} />
    <text x={260} y={y + 7} fill={C.cream} fontFamily={F.serif} fontStyle="italic" fontSize={24} textAnchor="middle">x</text>
    <line x1={280} y1={y} x2={330} y2={y} stroke={C.dim} strokeWidth={2} />
    {BOXES.map((b, i) => (
      <g key={i}>
        <rect x={b.x} y={y - BH / 2} width={BW} height={BH} rx={8} fill="rgba(8,14,32,0.9)" stroke={residual && fHi > 0 ? C.cyan : C.line}
          strokeWidth={residual ? lerp(1.5, 2.5, fHi) : 1.5} />
        <text x={b.x + BW / 2} y={y + 7} fill={C.cream} fontFamily={F.mono} fontSize={20} textAnchor="middle">{b.t}</text>
        {i < 4 && <line x1={b.x + BW} y1={y} x2={b.x + 160} y2={y} stroke={C.dim} strokeWidth={2} />}
      </g>
    ))}
    <line x1={1100} y1={y} x2={residual ? 1128 : 1210} y2={y} stroke={C.dim} strokeWidth={2} />
    {residual && (
      <g>
        <circle cx={1150} cy={y} r={22} fill={C.bg} stroke={C.amber} strokeWidth={2.5} opacity={shortcut > 0.95 ? 1 : 0.3} />
        <text x={1150} y={y + 9} fill={C.amber} fontFamily={F.sans} fontSize={30} textAnchor="middle" opacity={shortcut > 0.95 ? 1 : 0.3}>+</text>
        <line x1={1172} y1={y} x2={1210} y2={y} stroke={C.dim} strokeWidth={2} />
        <DrawPath d={`M260,${y + 20} L260,${y + 110} L1150,${y + 110} L1150,${y + 22}`} p={shortcut} stroke={C.amber} width={3} />
      </g>
    )}
    <rect x={1210} y={y - BH / 2} width={BW} height={BH} rx={8} fill="rgba(8,14,32,0.9)" stroke={C.line} strokeWidth={1.5} />
    <text x={1210 + BW / 2} y={y + 7} fill={C.cream} fontFamily={F.mono} fontSize={20} textAnchor="middle">ReLU</text>
    <Arrow x1={1340} y1={y} x2={1420} y2={y} color={C.dim} />
    <text x={1440} y={y + 8} fill={C.cream} fontFamily={F.serif} fontStyle="italic" fontSize={26}>y</text>
  </g>
);

export const Residual: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3, 4, 5].map(seg);

  const blockO = window(f, s[0] - 6, s[3] + 4, 16);
  const shortcut = prog(f, word(0, 'shortcut') - 2, 40);
  const fHi = prog(f, s[1] + 4, 16);
  const eqO = window(f, s[1] - 2, s[3] + 4, 16);
  const plainRowO = 1 - prog(f, s[2] - 6, 16);
  const padO = window(f, s[2] + 4, s[3] + 4, 16);

  const whyO = window(f, s[3] + 2, s[4] + 4, 16);
  const vec = prog(f, s[3] + 10, 30);
  const grad = linear(f, word(3, 'backprop') - 4, 90);

  const chartO = prog(f, s[4] - 4, 16);
  const shrink = prog(f, s[5], 30);
  const resReveal = [0, prog(f, word(4, '83.77') - 6, 14), prog(f, word(4, 'Eight'), 14), prog(f, word(4, 'Sixteen'), 14), prog(f, word(4, 'Thirty'), 14)];
  const cx = lerp(380, 150, shrink), cw = lerp(1160, 780, shrink);
  const notes = [
    {t: word(5, 'matters') - 6, c: C.cyan, body: <>At 32 conv: <b>{dep('plain', 32).mean.toFixed(2)}</b> → <b>{dep('residual', 32).mean.toFixed(2)}</b> (+{(dep('residual', 32).mean - dep('plain', 32).mean).toFixed(2)} pp)</>},
    {t: word(5, 'didnt') - 4, c: C.cream, body: <>At 4, 8, 16: residual did not beat plain ({[4, 8, 16].map((d) => `${dep('residual', d).mean.toFixed(2)} vs ${dep('plain', d).mean.toFixed(2)}`).join(' · ')})</>},
    {t: word(5, '0.03') - 4, c: C.amber, body: <>Winner residual 32 ({dep('residual', 32).mean.toFixed(2)} ± {dep('residual', 32).sd.toFixed(2)}) vs plain 16 ({dep('plain', 16).mean.toFixed(2)} ± {dep('plain', 16).sd.toFixed(2)}): Δ 0.03 pp, inside seed spread</>},
  ];

  return (
    <AbsoluteFill>
      <Abs x={0} y={0} o={blockO}>
        <Svg>
          <BlockRow y={420} residual={false} shortcut={0} fHi={0} label="plain" o={plainRowO * (1 - 0.3 * fHi)} />
          <BlockRow y={640} residual shortcut={shortcut} fHi={fHi} label="residual" />
          <g opacity={fHi}>
            <path d={`M330,595 L330,580 L1100,580 L1100,595`} stroke={C.cyan} strokeWidth={2} fill="none" />
            <text x={715} y={566} fill={C.cyan} fontFamily={F.serif} fontStyle="italic" fontSize={30} textAnchor="middle">F(x)</text>
          </g>
          <text x={705} y={782} fill={C.amber} fontFamily={F.mono} fontSize={20} textAnchor="middle" opacity={shortcut}>shortcut(x)</text>
        </Svg>
        <Tag x={1210} y={800} kind="code" o={prog(f, s[0] + 40, 14)}>../depth_cnn.py Block.forward</Tag>
      </Abs>

      <Eq x={960} y={232} size={54} o={eqO}>
        <V>y</V> = ReLU( <span style={{color: C.cyan}}><V>F</V>(<V>x</V>)</span> + <span style={{color: C.amber}}>shortcut(<V>x</V>)</span> )
      </Eq>

      {/* Zero-padded shortcut channels */}
      <Abs x={0} y={0} o={padO}>
        <Slab x={300} y={360} size={100} depth={60} layers={10} color={C.cyan} />
        <Txt x={300} y={480} w={260} size={20} font="mono" color={C.cyan}>x: 32 channels</Txt>
        <Txt x={520} y={378} w={60} size={40} color={C.muted}>+</Txt>
        <div style={{position: 'absolute', left: 600, top: 360, opacity: prog(f, s[2] + 20, 14)}}>
          <Slab x={0} y={0} size={100} depth={60} layers={10} color={C.dim} />
          <div style={{position: 'absolute', left: 30, top: 38, fontFamily: F.mono, fontSize: 22, color: C.muted}}>0</div>
        </div>
        <Txt x={600} y={480} w={300} size={20} font="mono" color={C.muted} o={prog(f, s[2] + 20, 14)}>zeros: 32 channels</Txt>
        <Txt x={830} y={378} w={60} size={40} color={C.muted} o={prog(f, s[2] + 34, 14)}>=</Txt>
        <Txt x={900} y={384} w={500} size={30} font="mono" o={prog(f, s[2] + 34, 14)}>64 channels</Txt>
        <Txt x={1220} y={360} w={600} size={20} font="mono" color={C.cream} o={prog(f, word(2, 'learned') - 4, 14)} lh={1.6}>
          torch.cat((x, x.new_zeros(...)), 1)<br />
          <span style={{color: C.muted}}>0 extra parameters → each residual net has</span><br />
          <span style={{color: C.muted}}>exactly its plain twin's count (1,239,274 at 32)</span>
        </Txt>
      </Abs>

      {/* Why it might help (schematic) */}
      <Abs x={0} y={0} o={whyO}>
        <Svg>
          <Arrow x1={260} y1={760} x2={700} y2={430} p={vec} color={C.cream} width={3} head={14} />
          <Arrow x1={700} y1={430} x2={790} y2={330} p={prog(f, s[3] + 40, 20)} color={C.cyan} width={3} head={12} />
          <Arrow x1={260} y1={760} x2={790} y2={330} p={prog(f, s[3] + 60, 24)} color={C.amber} width={2.5} head={14} dash="8 7" />
        </Svg>
        <Txt x={500} y={630} w={200} size={34} font="serif" italic o={vec}>x</Txt>
        <Txt x={770} y={400} w={360} size={30} font="serif" italic color={C.cyan} o={prog(f, s[3] + 40, 14)}>F(x): small correction</Txt>
        <Txt x={420} y={420} w={400} size={30} font="serif" italic color={C.amber} o={prog(f, s[3] + 60, 14)}>y = x + F(x)</Txt>
        <Txt x={260} y={800} w={700} size={21} font="mono" color={C.muted} o={prog(f, word(3, 'zero') - 4, 14)}>
          “do almost nothing” = push F toward 0, not rebuild x
        </Txt>
        <Svg>
          {[0, 1, 2, 3].map((k) => {
            const bx = 1060 + k * 190;
            return (
              <g key={k} opacity={prog(f, word(3, 'backprop') - 10, 14)}>
                <rect x={bx} y={560} width={120} height={70} rx={8} fill="rgba(8,14,32,0.9)" stroke={C.line} strokeWidth={1.5} />
                <text x={bx + 60} y={603} fill={C.muted} fontFamily={F.mono} fontSize={18} textAnchor="middle">F</text>
                <path d={`M${bx - 30},595 L${bx - 30},500 L${bx + 150},500 L${bx + 150},595`} stroke={C.amber} strokeWidth={2} fill="none" opacity={0.75} />
                <line x1={bx - 35} y1={595} x2={bx} y2={595} stroke={C.dim} strokeWidth={2} />
              </g>
            );
          })}
          {Array.from({length: 10}).map((_, i) => {
            const t = (grad * 1.6 + rand(i) * 0.6) % 1;
            if (grad <= 0) return null;
            const x = lerp(1840, 1030, t);
            const viaShortcut = i % 2 === 0;
            return <circle key={i} cx={x} cy={viaShortcut ? 500 : 595} r={viaShortcut ? 6 : 4} fill={C.amber} opacity={viaShortcut ? 0.95 : 0.45} />;
          })}
        </Svg>
        <Txt x={1040} y={660} w={760} size={20} font="mono" color={C.muted} o={prog(f, word(3, 'backprop'), 14)}>
          ← gradients flow back via shortcuts and through blocks
        </Txt>
        <Tag x={1040} y={440} kind="schematic" o={prog(f, word(3, 'backprop') - 10, 14)}>gradient routes · not measured from our runs</Tag>
        <Tag x={260} y={300} kind="schematic" o={vec}>vector picture of x + F(x) · not measured</Tag>
      </Abs>

      {/* Results */}
      <Abs x={0} y={0} o={chartO}>
        <Svg>
          <DepthChart x={cx} y={300} w={cw} h={440} plain={[1, 1, 1, 1, 1]} residual={resReveal} dimPlain={0.2} labels={1 - shrink * 0.4} />
        </Svg>
        <Tag x={cx} y={250} kind="validation">
          <span style={{color: C.cream}}>plain</span> vs <span style={{color: C.cyan}}>residual</span> · matched seeds and initial weights
        </Tag>
        <Abs x={1010} y={300} w={820} o={shrink}>
          {notes.map((n, i) => (
            <div key={i} style={{fontFamily: F.sans, fontSize: 24, color: C.cream, lineHeight: 1.45, marginBottom: 26, paddingLeft: 18,
              borderLeft: `3px solid ${n.c}`, opacity: prog(f, n.t, 14)}}>
              {n.body}
            </div>
          ))}
        </Abs>
        <Panel x={1010} y={680} w={780} h={130} o={prog(f, word(5, 'test') - 4, 16)} border={C.amber}>
          <div style={{padding: '18px 26px', fontFamily: F.sans, color: C.cream}}>
            <div style={{fontFamily: F.mono, fontSize: 17, color: C.amber, letterSpacing: 1.4}}>TEST · selected by validation · 3 seeds</div>
            <div style={{fontFamily: F.serif, fontSize: 52}}>{R.depth.test_mean.toFixed(2)}% <span style={{fontSize: 30, color: C.muted}}>± {R.depth.test_sd.toFixed(2)}</span></div>
          </div>
        </Panel>
      </Abs>
    </AbsoluteFill>
  );
};
