import React from 'react';
import {AbsoluteFill, Audio, Sequence, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {C, F, W} from './theme';
import {CHAPTERS, ChapterProvider, TOTAL_FRAMES, TRANSITION, chapterFrames, toFrame, Chapter} from './lib/time';
import {prog} from './lib/anim';
import {SCENES} from './scenes';
import './fonts';

const Background: React.FC = () => (
  <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 38%, ${C.bg2} 0%, ${C.bg} 62%, #03060f 100%)`}}>
    <svg width="100%" height="100%" style={{position: 'absolute', opacity: 0.35}}>
      <defs>
        <pattern id="dots" width="48" height="48" patternUnits="userSpaceOnUse">
          <circle cx="24" cy="24" r="1" fill={C.line} />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#dots)" />
    </svg>
  </AbsoluteFill>
);

const ChapterHeader: React.FC<{index: number; chapter: Chapter; duration: number}> = ({index, chapter, duration}) => {
  const f = useCurrentFrame();
  const o = Math.min(prog(f, 6, 20), 1 - prog(f, duration - 8, 12));
  return (
    <div style={{position: 'absolute', left: 80, top: 46, display: 'flex', alignItems: 'baseline', gap: 18, opacity: o}}>
      <span style={{fontFamily: F.mono, fontSize: 20, color: C.amber, letterSpacing: 2}}>{String(index + 1).padStart(2, '0')}</span>
      <span style={{fontFamily: F.serif, fontSize: 30, color: C.cream}}>{chapter.title}</span>
      <span style={{display: 'inline-block', width: 120 * prog(f, 10, 30), height: 1, background: C.line, alignSelf: 'center'}} />
    </div>
  );
};

const Progress: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <div style={{position: 'absolute', left: 0, top: 0, width: W, height: 3, background: 'rgba(36,52,94,0.35)'}}>
      <div style={{width: (W * f) / TOTAL_FRAMES, height: 3, background: C.cyanDim}} />
      {CHAPTERS.map((c) => (
        <div key={c.id} style={{position: 'absolute', left: (W * toFrame(c.start)) / TOTAL_FRAMES, top: 0, width: 2, height: 7, background: C.line}} />
      ))}
    </div>
  );
};

const Captions: React.FC = () => {
  const f = useCurrentFrame();
  const t = f / 30;
  let active: {text: string; start: number; end: number} | null = null;
  for (const ch of CHAPTERS)
    for (const s of ch.segments)
      for (const c of s.captions) if (t >= c.start && t < c.end) active = c;
  if (!active) return null;
  const o = Math.min(
    interpolate(t, [active.start, active.start + 0.12], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    interpolate(t, [active.end - 0.1, active.end], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
  );
  return (
    <div style={{position: 'absolute', left: 0, width: W, top: 972, display: 'flex', justifyContent: 'center', opacity: o}}>
      <div
        style={{
          maxWidth: 1560, padding: '9px 24px 11px', borderRadius: 8, background: 'rgba(3,7,18,0.78)',
          fontFamily: F.sans, fontSize: 33, color: C.cream, textAlign: 'center', lineHeight: 1.3, fontWeight: 400,
        }}
      >
        {active.text}
      </div>
    </div>
  );
};

export const Film: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: C.bg}}>
      <Background />
      {CHAPTERS.map((ch, i) => {
        const {from, duration} = chapterFrames(ch);
        const Scene = SCENES[ch.id];
        const last = i === CHAPTERS.length - 1;
        const span = duration + (last ? 0 : TRANSITION);
        return (
          <Sequence key={ch.id} from={from} durationInFrames={span} name={`${i + 1}. ${ch.title}`}>
            <ChapterProvider value={ch}>
              <ChapterLayer first={i === 0} last={last} span={span}>
                <Scene />
                <ChapterHeader index={i} chapter={ch} duration={duration} />
              </ChapterLayer>
            </ChapterProvider>
          </Sequence>
        );
      })}
      {CHAPTERS.flatMap((ch) =>
        ch.segments.map((s) => (
          <Sequence key={s.id} from={toFrame(s.start)} durationInFrames={toFrame(s.duration) + 15} name={`audio ${s.id}`}>
            <Audio src={staticFile(s.audio)} />
          </Sequence>
        )),
      )}
      <Captions />
      <Progress />
    </AbsoluteFill>
  );
};

const ChapterLayer: React.FC<{first: boolean; last: boolean; span: number; children: React.ReactNode}> = ({first, last, span, children}) => {
  const f = useCurrentFrame();
  const fadeIn = first ? prog(f, 0, 20) : prog(f, 0, TRANSITION);
  const fadeOut = last ? 1 - prog(f, span - 40, 40) : 1 - prog(f, span - TRANSITION, TRANSITION);
  return <AbsoluteFill style={{opacity: Math.min(fadeIn, fadeOut)}}>{children}</AbsoluteFill>;
};
