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

test('accepts 9 and 11 min gaps, interpolating at +5 min', () => {
  const late = fillMidpoints([
    ['2026-10-13T12:00', 1000, 20],
    ['2026-10-13T12:11', 2100, 9],
  ]);
  assert.deepEqual(late[1], ['2026-10-13T12:05', 1500, 15, true]);
  const early = fillMidpoints([
    ['2026-10-13T12:00', 900, 20],
    ['2026-10-13T12:09', 1800, 11],
  ]);
  assert.deepEqual(early[1], ['2026-10-13T12:05', 1400, 15, true]);
});

test('no midpoint for 8 or 12 min gaps', () => {
  for (const end of ['12:08', '12:12']) {
    const out = fillMidpoints([
      ['2026-10-13T12:00', 1000, 20],
      [`2026-10-13T${end}`, 1200, 18],
    ]);
    assert.equal(out.length, 2);
  }
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

test('20-min points get three 5-min points; older 10-min history keeps its midpoint', () => {
  const out = fillMidpoints(
    [
      ['2026-10-24T11:50', 1000, 20],
      ['2026-10-24T12:00', 1200, 18],
      ['2026-10-24T12:20', 1600, 14],
    ],
    { gapMin: 20 }
  );
  assert.deepEqual(
    out.map((p) => [p[0], p[1], p[3]]),
    [
      ['2026-10-24T11:50', 1000, false],
      ['2026-10-24T11:55', 1100, true],
      ['2026-10-24T12:00', 1200, false],
      ['2026-10-24T12:05', 1300, true],
      ['2026-10-24T12:10', 1400, true],
      ['2026-10-24T12:15', 1500, true],
      ['2026-10-24T12:20', 1600, false],
    ]
  );
});

test('with gapMin 20, a 21-min gap is filled and a 25-min gap is not', () => {
  const a = fillMidpoints([['2026-10-24T12:00', 1000, 20], ['2026-10-24T12:21', 2050, 10]], { gapMin: 20 });
  assert.equal(a.length, 5);
  const b = fillMidpoints([['2026-10-24T12:00', 1000, 20], ['2026-10-24T12:25', 2000, 10]], { gapMin: 20 });
  assert.equal(b.length, 2);
});
