import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fillMidpoints } from '../snapshot.js';

test('adds a midpoint between real points 10 min apart', () => {
  const out = fillMidpoints([
    ['2026-10-13T12:00', 1800, 14.0],
    ['2026-10-13T12:10', 2001, 13.2],
  ]);
  assert.deepEqual(out, [
    ['2026-10-13T12:00', 1800, 14.0, false],
    ['2026-10-13T12:05', 1901, 13.6, true],
    ['2026-10-13T12:10', 2001, 13.2, false],
  ]);
});

test('leaves 5-min history and longer gaps alone', () => {
  const out = fillMidpoints([
    ['2026-10-13T11:50', 1000, 20],
    ['2026-10-13T11:55', 1100, 19],
    ['2026-10-13T12:20', 1200, 18],
  ]);
  assert.equal(out.length, 3);
  assert.ok(out.every((p) => p[3] === false));
});

test('crosses midnight and keeps the newest point real', () => {
  const out = fillMidpoints([
    ['2026-10-13T23:55', 600, 30],
    ['2026-10-14T00:05', 700, 28],
  ]);
  assert.equal(out[1][0], '2026-10-14T00:00');
  assert.equal(out.at(-1)[3], false);
});

test('null speed stays null', () => {
  const out = fillMidpoints([
    ['2026-10-13T12:00', 1800, null],
    ['2026-10-13T12:10', 2000, 13],
  ]);
  assert.deepEqual(out[1], ['2026-10-13T12:05', 1900, null, true]);
});

test('empty input', () => {
  assert.deepEqual(fillMidpoints([]), []);
});
