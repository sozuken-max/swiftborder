// Pure helpers for the TomTom Routing API; no network or Google client imports, so tests run without npm install.

export const TOMTOM_ROUTING_URL = 'https://api.tomtom.com/routing/1/calculateRoute';

// Live-traffic car route leaving now. computeTravelTimeFor=all adds noTrafficTravelTimeInSeconds.
export function tomtomUrl(route, key) {
  const loc = (p) => `${p.lat},${p.lng}`;
  const params = new URLSearchParams({
    key,
    traffic: 'true',
    travelMode: 'car',
    routeType: 'fastest',
    departAt: 'now',
    computeTravelTimeFor: 'all',
  });
  return `${TOMTOM_ROUTING_URL}/${loc(route.origin)}:${loc(route.destination)}/json?${params}`;
}

// Fill a travel_times row from a calculateRoute reply. Same columns as the Distance Matrix row:
//   duration_in_traffic_sec = travelTimeInSeconds (live traffic)
//   duration_sec            = noTrafficTravelTimeInSeconds (free flow; Google's `duration` is a
//                             typical-traffic time, so congestion_ratio is not comparable across providers)
export function applyTomTomResponse(row, httpStatus, body) {
  const out = { ...row };

  if (httpStatus !== 200) {
    out.status = `HTTP_${httpStatus}`;
    const msg = body?.detailedError?.message || body?.error?.description || null;
    out.error_message = msg ? String(msg).slice(0, 1000) : null;
    return out;
  }

  const summary = body?.routes?.[0]?.summary;
  if (!summary) {
    out.status = 'NO_ROUTE';
    return out;
  }

  const distanceM = summary.lengthInMeters ?? null;
  const trafficSec = summary.travelTimeInSeconds ?? null;
  const freeFlowSec = summary.noTrafficTravelTimeInSeconds ?? null;

  out.status = trafficSec == null ? 'NO_TRAVEL_TIME' : 'OK';
  out.distance_m = distanceM;
  out.duration_sec = freeFlowSec;
  out.duration_in_traffic_sec = trafficSec;
  out.congestion_ratio =
    trafficSec && freeFlowSec ? +(trafficSec / freeFlowSec).toFixed(4) : null;
  out.speed_kmh =
    distanceM && trafficSec ? +((distanceM / 1000) / (trafficSec / 3600)).toFixed(2) : null;
  return out;
}
