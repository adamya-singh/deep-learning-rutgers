import {Config} from '@remotion/cli/config';

// CPU-only rendering: the GPU is reserved for the training queue.
Config.setVideoImageFormat('jpeg');
Config.setJpegQuality(92);
Config.setChromiumOpenGlRenderer('swiftshader');
Config.setConcurrency(4);
Config.setOverwriteOutput(true);
