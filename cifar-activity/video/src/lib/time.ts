import React, {createContext, useContext} from 'react';
import timeline from '../data/timeline.json';

export const FPS = timeline.fps;
export const TOTAL_FRAMES = timeline.durationInFrames;
export const TRANSITION = 14; // frames of cross-dissolve between chapters

export type Caption = {text: string; start: number; end: number};
export type Segment = {
  id: string;
  audio: string;
  start: number;
  duration: number;
  text: string;
  words: [string, number][];
  captions: Caption[];
};
export type Chapter = {id: string; title: string; start: number; end: number; segments: Segment[]};

export const CHAPTERS = timeline.chapters as unknown as Chapter[];
export const toFrame = (sec: number) => Math.round(sec * FPS);

export const chapterFrames = (c: Chapter) => ({
  from: toFrame(c.start),
  duration: toFrame(c.end) - toFrame(c.start),
});

const ChapterCtx = createContext<Chapter | null>(null);
export const ChapterProvider = ChapterCtx.Provider;

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9.]/g, '').replace(/\.$/, '');

/**
 * Cue helpers for the current chapter. All values are frames relative to the
 * chapter start, which is what useCurrentFrame() returns inside its Sequence.
 */
export const useCues = () => {
  const ch = useContext(ChapterCtx);
  if (!ch) throw new Error('useCues outside a chapter');
  const base = ch.start;
  const seg = (i: number) => toFrame(ch.segments[i].start - base);
  const segEnd = (i: number) => toFrame(ch.segments[i].start + ch.segments[i].duration - base);
  /** Frame at which the n-th token starting with `prefix` is spoken in segment i. */
  const word = (i: number, prefix: string, nth = 0) => {
    const p = norm(prefix);
    const hits = ch.segments[i].words.filter(([w]) => norm(w).startsWith(p));
    const hit = hits[Math.min(nth, hits.length - 1)];
    if (!hit) throw new Error(`word "${prefix}" not found in ${ch.segments[i].id}`);
    return toFrame(hit[1] - base);
  };
  const end = toFrame(ch.end) - toFrame(ch.start);
  return {seg, segEnd, word, end, chapter: ch};
};

export const h = React.createElement;
