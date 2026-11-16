import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

test('a second camera click creates a point on a plain HTTP lab page', () => {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) {
      elements.set(id, {addEventListener() {}, getContext() {}});
    }
    return elements.get(id);
  };
  element('mode').value = 'point';
  element('guide').checked = false;

  const context = vm.createContext({
    // Plain HTTP on a Jetson IP exposes getRandomValues but not randomUUID.
    crypto: {getRandomValues: webcrypto.getRandomValues.bind(webcrypto)},
    document: {body: {}, getElementById: element},
    fetch: () => new Promise(() => {}),
    ResizeObserver: class {observe() {}},
    window: {addEventListener() {}},
  });
  const source = readFileSync(new URL('../wave_analysis/static/annotate.js', import.meta.url), 'utf8');
  vm.runInContext(source, context);
  const points = JSON.parse(vm.runInContext(`
    draw = () => {};
    renderObjects = () => {};
    updateGeometry = () => {};
    documentData = {points: []};
    addClick(0, [10, 20]);
    addClick(1, [30, 40]);
    addClick(1, [31, 41]);
    addClick(0, [11, 21]);
    JSON.stringify(documentData.points);
  `, context));

  assert.deepEqual(points.map(({cam0, cam1, status}) => ({cam0, cam1, status})), [
    {cam0: [10, 20], cam1: [30, 40], status: 'uncertain'},
    {cam0: [11, 21], cam1: [31, 41], status: 'uncertain'},
  ]);
  assert.match(points[0].id, /^[a-f0-9]{32}$/);
  assert.match(points[1].id, /^[a-f0-9]{32}$/);
  assert.notEqual(points[0].id, points[1].id);
});
