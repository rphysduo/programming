import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function loadEditor() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) {
      const listeners = {};
      elements.set(id, {
        listeners,
        style: {},
        addEventListener(type, handler) { listeners[type] = handler; },
        getContext() {},
        getBoundingClientRect() { return {left:10, top:20, width:640, height:360}; },
      });
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
    window: {innerHeight:350, addEventListener() {}},
  });
  const source = readFileSync(new URL('../wave_analysis/static/annotate.js', import.meta.url), 'utf8');
  vm.runInContext(source, context);
  return {context, element};
}

test('a second camera click creates a point on a plain HTTP lab page', () => {
  const {context} = loadEditor();
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

test('magnifier centers the raw cursor pixel after zoom and pan, and hides off the image', () => {
  const {context, element} = loadEditor();
  const canvas = element('view1');
  const lens = element('lens1');
  const draws = [];
  context.lensContext = {
    setTransform() {}, clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    drawImage(...args) { draws.push(args); },
  };
  element('magnify').checked = true;
  vm.runInContext(`
    const view = views[1];
    view.image = {width:1920, height:1080};
    view.scale = 2;
    view.offset = [-100, -80];
    view.lensContext = lensContext;
  `, context);

  canvas.listeners.pointermove({clientX:610, clientY:320});
  assert.equal(lens.hidden, false);
  assert.equal(draws.length, 1);
  const [image, imageLeft, imageTop, imageWidth, imageHeight] = draws[0];
  const center = parseFloat(lens.style.width) / 2;
  assert.equal((center - imageLeft) / (imageWidth / image.width), 350);
  assert.equal((center - imageTop) / (imageHeight / image.height), 190);
  assert.ok(parseFloat(lens.style.left) + 2 * center < 600);
  assert.ok(parseFloat(lens.style.top) + 2 * center < 300);

  canvas.listeners.pointerleave();
  assert.equal(lens.hidden, true);
  canvas.listeners.pointermove({clientX:8, clientY:30});
  assert.equal(lens.hidden, true);
  canvas.listeners.pointermove({clientX:80, clientY:90});
  assert.equal(lens.hidden, false);
  element('magnify').checked = false;
  element('magnify').onchange();
  canvas.listeners.pointermove({clientX:80, clientY:90});
  assert.equal(lens.hidden, true);
});
