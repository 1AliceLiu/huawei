const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Exercise the editor's event handlers and outgoing requests without a browser dependency.
async function editor({saved = null, noFrame = false, saveError = null} = {}) {
  const elements = new Map();
  const drawing = new Proxy({}, {get: () => () => {}});
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      dataset: {}, disabled: true, textContent: '', listeners: {},
      addEventListener(name, callback) {this.listeners[name] = callback;},
      setAttribute() {},
      click() {if (!this.disabled) return this.listeners.click?.();},
      getContext: () => drawing,
      getBoundingClientRect: () => ({left: 10, top: 20, width: 640, height: 360}),
    });
    return elements.get(id);
  }
  const buttons = ['roi', 'line', 'direction'].map(mode => {
    const button = element(mode);
    button.dataset.mode = mode;
    return button;
  });
  const posts = [];
  const context = vm.createContext({
    document: {getElementById: element, querySelectorAll: () => buttons, addEventListener() {}},
    window: {addEventListener() {}}, location: {pathname: '/calibrate/0'},
    URL: {createObjectURL: () => 'blob:test-frame', revokeObjectURL() {}},
    Image: class {naturalWidth = 1280; naturalHeight = 720; async decode() {}},
    fetch: async (url, options = {}) => {
      if (options.method === 'POST') {
        posts.push({url, data: JSON.parse(options.body)});
        return {ok: !saveError, json: async () => saveError ? {error: saveError} : {saved: true}};
      }
      if (url.startsWith('/calibration-frame/')) return {ok: !noFrame, blob: async () => ({})};
      return {ok: true, json: async () => ({camera: '2763', calibration: saved})};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/calibration.js'), 'utf8'), context);
  await new Promise(setImmediate);
  return {
    element, posts,
    point(x, y) {element('canvas').listeners.pointerdown({button: 0, clientX: 10 + 640 * x, clientY: 20 + 360 * y});},
  };
}

function rectangle(ui) {
  for (const point of [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]) ui.point(...point);
  ui.element('close-roi').click();
}

test('scaled canvas clicks, closure, undo and direction produce normalized saved coordinates', async () => {
  const ui = await editor();
  rectangle(ui);
  ui.point(.2, .5);
  ui.point(.8, .5);
  ui.point(.5, .3);
  ui.point(.4, .7);
  ui.element('undo').click();
  ui.point(.5, .7);
  await ui.element('save').click();
  assert.deepEqual(ui.posts, [{url: '/api/calibration/0', data: {
    roi: [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
    line: [[.2, .5], [.8, .5]], direction: [[.5, .3], [.5, .7]], frame_size: [1280, 720],
  }}]);
  assert.equal(ui.element('status').dataset.error, 'false');
});

test('ROI can be closed by clicking its first vertex and saved without a count line', async () => {
  const ui = await editor();
  for (const point of [[.1, .1], [.9, .1], [.9, .9], [.1, .9], [.1, .1]]) ui.point(...point);
  await ui.element('save').click();
  assert.equal(ui.posts[0].data.roi.length, 4);
  assert.deepEqual(ui.posts[0].data.line, []);
});

test('unfinished ROI never issues a save request', async () => {
  const ui = await editor();
  ui.point(.1, .1);
  ui.point(.9, .1);
  await ui.element('save').click();
  assert.equal(ui.posts.length, 0);
  assert.equal(ui.element('status').dataset.error, 'true');
});

test('loading saved geometry preserves it and clear remains undoable', async () => {
  const saved = {roi: [[.1, .1], [.9, .1], [.5, .9]], line: [], direction: [], frame_size: [1280, 720]};
  const ui = await editor({saved});
  ui.element('clear').click();
  ui.element('undo').click();
  await ui.element('save').click();
  assert.deepEqual(ui.posts[0].data, saved);
  ui.element('clear').click();
  await ui.element('save').click();
  assert.deepEqual(ui.posts[1].data, {roi: [], line: [], direction: [], frame_size: [1280, 720]});
});

test('failed saves keep edits available for a retry and surface the server error', async () => {
  const ui = await editor({saveError: 'disk full'});
  rectangle(ui);
  await ui.element('save').click();
  assert.equal(ui.element('status').textContent, 'disk full');
  assert.equal(ui.element('save').disabled, false);
  await ui.element('save').click();
  assert.deepEqual(ui.posts[1], ui.posts[0]);
});

test('a missing camera frame disables editing and saving', async () => {
  const ui = await editor({noFrame: true});
  assert.equal(ui.element('status').dataset.error, 'true');
  assert.equal(ui.element('save').disabled, true);
  assert.equal(ui.posts.length, 0);
});
