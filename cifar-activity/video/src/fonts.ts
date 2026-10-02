import {loadFont} from '@remotion/fonts';
import {staticFile} from 'remotion';

const GREEK = 'U+0370-03FF, U+1F00-1FFF';

const faces: {family: string; file: string; weight: string; style?: string; unicodeRange?: string}[] = [
  {family: 'Inter', file: 'inter-latin-400-normal.woff2', weight: '400'},
  {family: 'Inter', file: 'inter-latin-500-normal.woff2', weight: '500'},
  {family: 'Inter', file: 'inter-latin-600-normal.woff2', weight: '600'},
  {family: 'Inter', file: 'inter-greek-400-normal.woff2', weight: '400', unicodeRange: GREEK},
  {family: 'Source Serif 4', file: 'source-serif-4-latin-400-normal.woff2', weight: '400'},
  {family: 'Source Serif 4', file: 'source-serif-4-latin-600-normal.woff2', weight: '600'},
  {family: 'Source Serif 4', file: 'source-serif-4-latin-400-italic.woff2', weight: '400', style: 'italic'},
  {family: 'Source Serif 4', file: 'source-serif-4-latin-600-italic.woff2', weight: '600', style: 'italic'},
  {family: 'Source Serif 4', file: 'source-serif-4-greek-400-normal.woff2', weight: '400', unicodeRange: GREEK},
  {family: 'Source Serif 4', file: 'source-serif-4-greek-400-italic.woff2', weight: '400', style: 'italic', unicodeRange: GREEK},
  {family: 'Source Serif 4', file: 'source-serif-4-greek-600-italic.woff2', weight: '600', style: 'italic', unicodeRange: GREEK},
  {family: 'JetBrains Mono', file: 'jetbrains-mono-latin-400-normal.woff2', weight: '400'},
  {family: 'JetBrains Mono', file: 'jetbrains-mono-latin-500-normal.woff2', weight: '500'},
];

export const fontsReady = Promise.all(
  faces.map((f) =>
    loadFont({
      family: f.family,
      url: staticFile(`fonts/${f.file}`),
      weight: f.weight,
      style: f.style ?? 'normal',
      unicodeRange: f.unicodeRange,
    }),
  ),
);
