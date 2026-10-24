import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CARRY_TAG, carryForward, parseInterval } from '../carry.js';

const ok = {
  observed_at: '2026-10-24T04:00:03.000Z',
  observed_at_sgt: '2026-10-24T12:00:03',
  observed_date_sgt: '2026-10-24',
  route_id: 'jb_to_woodlands',
  direction: 'MY_TO_SG',
  status: 'OK',
  duration_in_traffic_sec: 1500,
  speed_kmh: 20.5,
  error_message: null,
};

test('20-min runs fill the skipped 10-min bin with the same reading', () => {
  const out = carryForward([ok], 20);
  assert.equal(out.length, 1);
  assert.equal(out[0].observed_at, '2026-10-24T04:10:03.000Z');
  assert.equal(out[0].observed_at_sgt, '2026-10-24T12:10:03');
  assert.equal(out[0].duration_in_traffic_sec, 1500);
  assert.equal(out[0].status, 'OK');
  assert.ok(out[0].error_message.startsWith(CARRY_TAG));
  assert.equal(ok.error_message, null); // input untouched
});

test('10-min runs add nothing; 30-min runs fill two bins', () => {
  assert.deepEqual(carryForward([ok], 10), []);
  assert.deepEqual(
    carryForward([ok], 30).map((r) => r.observed_at),
    ['2026-10-24T04:10:03.000Z', '2026-10-24T04:20:03.000Z']
  );
});

test('failed rows are not carried', () => {
  assert.deepEqual(carryForward([{ ...ok, status: 'ZERO_RESULTS' }], 20), []);
});

test('carried rows cross SGT midnight', () => {
  const late = { ...ok, observed_at: '2026-10-24T15:55:00.000Z' };
  assert.equal(carryForward([late], 20)[0].observed_date_sgt, '2026-10-25');
});

test('interval defaults to 20 and must be a multiple of 10', () => {
  assert.equal(parseInterval(undefined), 20);
  assert.equal(parseInterval('10'), 10);
  for (const bad of ['5', '15', '0', 'x']) assert.throws(() => parseInterval(bad));
});
