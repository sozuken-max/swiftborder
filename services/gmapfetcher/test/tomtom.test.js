import { test } from 'node:test';
import assert from 'node:assert/strict';
import { applyTomTomResponse, tomtomUrl } from '../tomtom.js';
import { parseFlag } from '../carry.js';

const route = {
  route_id: 'jb_to_woodlands',
  origin: { lat: 1.4884465059461982, lng: 103.78625911225137 },
  destination: { lat: 1.4401375461664851, lng: 103.76842889302142 },
};

const base = {
  route_id: 'jb_to_woodlands',
  provider: 'tomtom',
  status: 'ERROR',
  distance_m: null,
  duration_sec: null,
  duration_in_traffic_sec: null,
  congestion_ratio: null,
  speed_kmh: null,
  error_message: null,
};

test('url puts origin:destination in the path and asks for live traffic', () => {
  const url = new URL(tomtomUrl(route, 'k3y'));
  assert.equal(
    url.pathname,
    '/routing/1/calculateRoute/1.4884465059461982,103.78625911225137:1.4401375461664851,103.76842889302142/json'
  );
  assert.equal(url.searchParams.get('key'), 'k3y');
  assert.equal(url.searchParams.get('traffic'), 'true');
  assert.equal(url.searchParams.get('travelMode'), 'car');
  assert.equal(url.searchParams.get('computeTravelTimeFor'), 'all');
  assert.equal(url.searchParams.has('vehicleHeading'), false);
});

test('url sends vehicleHeading only when the route sets one', () => {
  const url = new URL(tomtomUrl({ ...route, tomtom_heading: 198 }, 'k3y'));
  assert.equal(url.searchParams.get('vehicleHeading'), '198');
  assert.equal(new URL(tomtomUrl({ ...route, tomtom_heading: 0 }, 'k3y')).searchParams.get('vehicleHeading'), '0');
});

test('OK reply fills the same columns as a Distance Matrix row', () => {
  const out = applyTomTomResponse(base, 200, {
    routes: [
      {
        summary: {
          lengthInMeters: 6000,
          travelTimeInSeconds: 1200,
          noTrafficTravelTimeInSeconds: 600,
          trafficDelayInSeconds: 600,
        },
      },
    ],
  });
  assert.equal(out.status, 'OK');
  assert.equal(out.provider, 'tomtom');
  assert.equal(out.distance_m, 6000);
  assert.equal(out.duration_in_traffic_sec, 1200);
  assert.equal(out.duration_sec, 600);
  assert.equal(out.congestion_ratio, 2);
  assert.equal(out.speed_kmh, 18);
  assert.equal(base.status, 'ERROR'); // input untouched
});

test('HTTP error keeps the row with the status and message', () => {
  const out = applyTomTomResponse(base, 403, {
    detailedError: { code: 'Forbidden', message: 'Developer Inactive' },
  });
  assert.equal(out.status, 'HTTP_403');
  assert.equal(out.error_message, 'Developer Inactive');
  assert.equal(out.duration_in_traffic_sec, null);

  const legacy = applyTomTomResponse(base, 400, { error: { description: 'Invalid request' } });
  assert.equal(legacy.error_message, 'Invalid request');

  assert.equal(applyTomTomResponse(base, 502, null).error_message, null);
});

test('200 without a route or travel time is not OK', () => {
  assert.equal(applyTomTomResponse(base, 200, { routes: [] }).status, 'NO_ROUTE');
  assert.equal(
    applyTomTomResponse(base, 200, { routes: [{ summary: { lengthInMeters: 6000 } }] }).status,
    'NO_TRAVEL_TIME'
  );
});

test('parseFlag accepts common spellings and rejects typos', () => {
  assert.equal(parseFlag('X', undefined, true), true);
  assert.equal(parseFlag('X', '', false), false);
  assert.equal(parseFlag('X', ' On ', false), true);
  assert.equal(parseFlag('X', '0', true), false);
  assert.equal(parseFlag('X', 'FALSE', true), false);
  assert.throws(() => parseFlag('X', 'flase', true), /X must be true or false/);
});
