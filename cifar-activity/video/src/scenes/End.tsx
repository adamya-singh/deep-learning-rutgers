import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, F} from '../theme';
import {useCues} from '../lib/time';
import {lerp, prog, window} from '../lib/anim';
import {Abs, Panel, Svg, Tag, Txt} from '../components/ui';
import {R, dep, fu} from '../lib/data';
import timeline from '../data/timeline.json';

const PRETTY: Record<string, string> = {
  'winner-none': 'residual 32 · none', 'winner-flip': 'residual 32 · flip', 'winner-crop-flip': 'residual 32 · crop + flip', 'resnet18-crop-flip': 'ResNet-18 · crop + flip',
};

export const End: React.FC = () => {
  const f = useCurrentFrame();
  const {seg, word, end} = useCues();
  const s = [0, 1, 2, 3, 4].map(seg);
  const rn = fu('resnet18-crop-flip');
  const selected = (R.followup.selection as {candidate?: string} | null)?.candidate ?? '';

  const statusO = window(f, s[0] - 6, s[1] + 4, 16);
  const jobs = Object.entries(R.followup.jobs);
  const groups = ['winner-flip', 'winner-crop-flip', 'resnet18-crop-flip'];

  const steps = [
    {v: R.classroom.baseline_b64[9].test, m: 'TEST', label: 'classroom baseline', sub: '2 conv · 10 epochs', cue: R.classroom.baseline_b64[9].test.toFixed(2)},
    {v: dep('plain', 2).mean, m: 'VAL', label: 'stricter recipe', sub: '2 conv + BN · 160 ep', cue: dep('plain', 2).mean.toFixed(2)},
    {v: dep('residual', 32).mean, m: 'VAL', label: 'depth + shortcuts', sub: 'residual 32', cue: dep('residual', 32).mean.toFixed(2)},
    {v: fu('winner-crop-flip').mean, m: 'VAL', label: '+ crop & flip', sub: 'residual 32', cue: fu('winner-crop-flip').mean.toFixed(2)},
    {v: rn.mean, m: 'VAL', label: 'ResNet-18', sub: R.followup.test_results_available && selected === 'resnet18-crop-flip' ? `3-seed val · TEST ${R.followup.test_mean?.toFixed(2)}` : `crop & flip · ${rn.seeds.length}/3 seeds`, cue: rn.mean.toFixed(2)},
  ];
  const pathO = window(f, s[1] - 4, s[3] + 4, 16);
  const dimPath = prog(f, s[2], 20);
  const bx = (i: number) => 250 + i * 300;
  const by = (v: number) => 780 - ((v - 50) / 50) * 440;
  const scopeO = window(f, s[2] + 4, s[3] + 4, 16);

  const lessons = [
    ['Compare runs by optimizer steps, not just epochs.', 'comparing'],
    ['Decide on validation, across seeds.', 'deciding'],
    ['A smaller train–test gap isn’t an improvement by itself.', 'smaller'],
    ['Retest a technique when the model or recipe changes.', 'retesting'],
  ];
  const lessonO = window(f, s[3] - 2, s[4] + 4, 16);
  const finalO = prog(f, s[4] - 4, 20);

  return (
    <AbsoluteFill>
      <Abs x={0} y={0} o={statusO}>
        <Txt x={300} y={250} w={1300} size={24} font="mono" color={C.muted}>
          follow-up queue at snapshot {timeline.snapshot_utc.replace('T', ' ').replace('Z', ' UTC')} · phase: {R.followup.phase}
        </Txt>
        {groups.map((g, r) => (
          <React.Fragment key={g}>
            <Txt x={300} y={330 + r * 90} w={420} size={26} font="mono">{PRETTY[g]}</Txt>
            {[0, 1, 2].map((sd) => {
              const st = R.followup.jobs[`${g}/seed-${sd}`] ?? 'pending';
              const ep = R.followup.running_epochs[`${g}/seed-${sd}`];
              const col = st === 'completed' ? C.cyan : st === 'running' ? C.amber : C.dim;
              return (
                <div key={sd} style={{position: 'absolute', left: 780 + sd * 270, top: 322 + r * 90, width: 250, height: 50, borderRadius: 8,
                  border: `1.5px solid ${col}`, color: col, fontFamily: F.mono, fontSize: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  opacity: prog(f, s[0] + 10 + r * 6 + sd * 2, 12)}}>
                  seed {sd} · {st === 'running' && ep ? `ep ${ep}/200` : st}
                </div>
              );
            })}
          </React.Fragment>
        ))}
        <Txt x={300} y={620} w={1300} size={24} font="mono" color={C.muted} o={prog(f, s[0] + 40, 14)}>
          residual 32 · none: reused from the depth sweep ({jobs.length} new jobs tracked)
        </Txt>
        <Txt x={300} y={680} w={1400} size={30} font="serif" o={prog(f, s[0] + 40, 14)}>
          {R.followup.test_results_available
            ? <>Selected by validation: <span style={{color: C.cyan}}>{PRETTY[selected] ?? selected}</span> · <span style={{color: C.amber}}>TEST</span> {R.followup.test_mean?.toFixed(2)}%{R.followup.test_sd ? ` ± ${R.followup.test_sd.toFixed(2)}` : ''}</>
            : <>Official selection: <span style={{color: C.amber}}>pending</span> · ResNet-18 test score: <span style={{color: C.amber}}>not available</span></>}
        </Txt>
        <Tag x={300} y={760} kind="code">../runs/followup/queue-state.json, comparison.json (frozen copies in video/data/frozen/)</Tag>
      </Abs>

      <Abs x={0} y={0} o={pathO * (1 - dimPath)}>
        <Svg>
          {[60, 70, 80, 90, 100].map((t) => (
            <g key={t}>
              <line x1={200} x2={1720} y1={by(t)} y2={by(t)} stroke={C.grid} />
              <text x={186} y={by(t) + 6} fill={C.muted} fontFamily={F.mono} fontSize={16} textAnchor="end">{t}</text>
            </g>
          ))}
          {steps.map((st, i) => {
            const p = prog(f, word(1, st.cue) - 6, 18);
            const col = st.m === 'TEST' ? C.amber : C.cyan;
            return (
              <g key={i} opacity={p}>
                <rect x={bx(i)} y={lerp(by(50), by(st.v), p)} width={180} height={by(50) - lerp(by(50), by(st.v), p)} fill={col} opacity={0.22} />
                <line x1={bx(i)} x2={bx(i) + 180} y1={by(st.v)} y2={by(st.v)} stroke={col} strokeWidth={4} />
                {i > 0 && <line x1={bx(i - 1) + 180} x2={bx(i)} y1={by(steps[i - 1].v)} y2={by(st.v)} stroke={C.dim} strokeWidth={1.5} strokeDasharray="5 5" />}
              </g>
            );
          })}
        </Svg>
        {steps.map((st, i) => {
          const p = prog(f, word(1, st.cue) - 6, 18);
          const col = st.m === 'TEST' ? C.amber : C.cyan;
          return (
            <React.Fragment key={i}>
              <Txt x={bx(i) + 90} y={by(st.v) - 62} w={260} align="center" size={34} font="serif" o={p}>{st.v.toFixed(2)}</Txt>
              <Txt x={bx(i) + 90} y={by(st.v) - 86} w={260} align="center" size={14} font="mono" color={col} o={p} style={{letterSpacing: 1.5}}>
                {st.m === 'TEST' ? 'TEST' : 'VALIDATION'}
              </Txt>
              <Txt x={bx(i) + 90} y={798} w={290} align="center" size={20} o={p}>{st.label}</Txt>
              <Txt x={bx(i) + 90} y={828} w={290} align="center" size={16} font="mono" color={C.muted} o={p}>{st.sub}</Txt>
            </React.Fragment>
          );
        })}
        <Tag x={960} y={244} kind="note" anchor="center" o={prog(f, word(1, 'leaderboard') - 6, 14)}>mixed metrics: the order things happened, not a leaderboard</Tag>
      </Abs>

      <Abs x={0} y={0} o={scopeO}>
        <Txt x={960} y={360} w={1500} align="center" size={50} font="serif">
          Strongest result <span style={{color: C.cyan}}>measured here</span>, not the best ResNet
        </Txt>
        <Txt x={960} y={470} w={1300} align="center" size={26} color={C.muted} lh={1.5} o={prog(f, word(2, 'Wider') - 4, 14)}>
          Wider and deeper residual networks and other training recipes report higher CIFAR-10 accuracy.
          These experiments make no state-of-the-art claim.
        </Txt>
        <Txt x={960} y={600} w={1300} align="center" size={18} font="mono" color={C.dim} o={prog(f, word(2, 'Wider'), 14)}>
          e.g. Wide Residual Networks, arxiv.org/abs/1605.07146
        </Txt>
      </Abs>

      <Abs x={0} y={0} o={lessonO}>
        {lessons.map(([t, cue], i) => (
          <div key={i} style={{position: 'absolute', left: 340, top: 290 + i * 120, width: 1400, whiteSpace: 'nowrap', display: 'flex', alignItems: 'baseline', gap: 30,
            opacity: prog(f, word(3, cue) - 6, 14), transform: `translateX(${lerp(-30, 0, prog(f, word(3, cue) - 6, 18))}px)`}}>
            <span style={{fontFamily: F.mono, fontSize: 26, color: C.amber}}>{String(i + 1).padStart(2, '0')}</span>
            <span style={{fontFamily: F.serif, fontSize: 46, color: C.cream}}>{t}</span>
          </div>
        ))}
      </Abs>

      <Abs x={0} y={0} o={finalO}>
        <Txt x={960} y={350} w={1500} align="center" size={72} font="serif">
          {R.classroom.baseline_b64[9].test.toFixed(2)}% <span style={{color: C.amber}}>→</span> {R.followup.test_results_available ? `${R.followup.test_mean?.toFixed(2)}%` : `${rn.mean.toFixed(2)}%`} <span style={{fontSize: 34, color: C.muted}}>{R.followup.test_results_available ? 'test' : 'validation'}</span>
        </Txt>
        <Txt x={960} y={460} w={1500} align="center" size={24} font="mono" color={C.muted} o={prog(f, s[4] + 20, 16)}>
          recipe · depth + shortcuts · augmentation · architecture · contributions not separated
        </Txt>
        <Panel x={460} y={560} w={1000} h={226} o={prog(f, s[4] + 30, 20)} border={C.line}>
          <div style={{padding: '22px 30px', fontFamily: F.mono, fontSize: 17, color: C.muted, lineHeight: 1.75}}>
            <div>data: CIFAR-10 (Krizhevsky, 2009) · cs.toronto.edu/~kriz/cifar.html</div>
            <div>evidence: cifar-activity runs/, cifar_cnn.py, depth_cnn.py, followup_cnn.py</div>
            <div>residual networks: He et al., arxiv.org/abs/1512.03385</div>
            <div>narration: synthetic neural voice (edge-tts, en-US-AndrewNeural)</div>
            <div>animated with Remotion · written and built with Claude Code (Claude Opus 5.5)</div>
            <div>results snapshot: {timeline.snapshot_utc}</div>
          </div>
        </Panel>
      </Abs>
    </AbsoluteFill>
  );
};
