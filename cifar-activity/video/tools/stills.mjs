// Render preview stills (bundle once, many frames). CPU only.
//   node tools/stills.mjs                 → one frame per narration segment (75% through it)
//   node tools/stills.mjs 1200 4500       → specific frames
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition, openBrowser} from '@remotion/renderer';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const timeline = JSON.parse(fs.readFileSync(path.join(root, 'src/data/timeline.json'), 'utf8'));
const outDir = path.join(root, 'out/frames');
fs.mkdirSync(outDir, {recursive: true});

let frames = process.argv.slice(2).map(Number);
const names = {};
if (frames.length === 0) {
  for (const ch of timeline.chapters) {
    ch.segments.forEach((s) => {
      const fr = Math.round((s.start + s.duration * 0.75) * timeline.fps);
      frames.push(fr);
      names[fr] = `${s.id}`;
    });
  }
}

const serveUrl = await bundle({entryPoint: path.join(root, 'src/index.ts')});
const browser = await openBrowser('chrome', {
  browserExecutable: '/usr/bin/google-chrome',
  chromiumOptions: {gl: 'swiftshader'},
});
const composition = await selectComposition({serveUrl, id: 'CifarJourney', puppeteerInstance: browser});
for (const frame of frames) {
  const name = `${String(frame).padStart(5, '0')}${names[frame] ? '-' + names[frame] : ''}.jpg`;
  try {
    await renderStill({
      composition, serveUrl, frame, output: path.join(outDir, name), imageFormat: 'jpeg', jpegQuality: 88,
      puppeteerInstance: browser, chromiumOptions: {gl: 'swiftshader'},
    });
    console.log('rendered', name);
  } catch (e) {
    console.log('FAILED', name, String(e.message).split('\n')[0]);
  }
}
await browser.close({silent: true});
