import React from 'react';
import {Task} from './Task';
import {Cnn} from './Cnn';
import {Learn} from './Learn';
import {Batch} from './Batch';
import {Aug1} from './Aug1';
import {Recipe} from './Recipe';
import {Depth} from './Depth';
import {Residual} from './Residual';
import {Aug2} from './Aug2';
import {ResNet} from './ResNet';
import {End} from './End';

export const SCENES: Record<string, React.FC> = {
  task: Task, cnn: Cnn, learn: Learn, batch: Batch, aug1: Aug1, recipe: Recipe, depth: Depth,
  residual: Residual, aug2: Aug2, resnet: ResNet, end: End,
};
