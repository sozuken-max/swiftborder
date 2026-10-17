"""The served mix is eval.layer_b_fcm_xgb_mix, not a second copy of the arithmetic."""

import datetime as dt

import numpy as np
import pandas as pd

import mix_bridge


def _fast_config():
    base = mix_bridge.mixmod.fcm.Config(
        ensemble_size=1,
        max_epochs=4,
        patience=2,
        mlp_alphas=(1e-3,),
        fcm_restarts=1,
        protected_window=("", ""),
    )
    return mix_bridge.mixmod.MixConfig(base=base, min_xgb_rows=100, history_start="2026-09-06")


def _bins():
    """Seven SGT days of both directions, plus one origin on the next morning."""
    start = pd.Timestamp("2026-09-12 16:00", tz="UTC")  # 2026-09-13 00:00 SGT
    periods = 7 * 144 + 6  # through 01:00 SGT on 20 Sep
    times = pd.date_range(start, periods=periods, freq="10min")
    rng = np.random.default_rng(0)
    rows = []
    for direction, shift in (("SG_TO_MY", 0.0), ("MY_TO_SG", 4.0)):
        for ts in times:
            hour = ts.tz_convert("Asia/Singapore").hour
            dur = 18.0 + shift + 6.0 * np.sin(2 * np.pi * hour / 24.0) + float(rng.normal(0, 0.05))
            rows.append(
                {
                    "direction": direction,
                    "bin_ts": ts,
                    "dur_min": dur,
                    "congestion_ratio": 1.1,
                    "speed_kmh": 40.0,
                }
            )
    frame = pd.DataFrame(rows)
    day_start = pd.Timestamp("2026-09-19 16:00", tz="UTC")  # 2026-09-20 00:00 SGT
    history = frame[frame["bin_ts"] < day_start].reset_index(drop=True)
    origin = pd.Timestamp("2026-09-19 16:50", tz="UTC")
    latest = frame[frame["bin_ts"] <= origin].reset_index(drop=True)
    return history, latest, day_start, origin


def test_combine_is_the_offline_mix():
    fcm = np.array([2.0, np.nan, 4.0])
    xgb = np.array([4.0, 6.0, 8.0])
    got = mix_bridge.combine(fcm, xgb)
    np.testing.assert_allclose(got, mix_bridge.mixmod.mix_of(fcm, xgb), equal_nan=True)
    assert mix_bridge.WEIGHTS == {"fcm_mlp": 0.5, "xgb[maps]": 0.5}
    assert mix_bridge.mixmod.MIX_WEIGHTS == {"fcm_mlp": 0.5, "xgb_maps": 0.5}


def test_predict_matches_fit_day_and_predict_direction():
    history, latest, day_start, origin = _bins()
    cfg = _fast_config()
    fitted = mix_bridge.fit_models(history, day_start, cfg=cfg, protect_labels=False)
    value, parts = mix_bridge.predict_value(fitted, history, latest, "SG_TO_MY", origin, cfg=cfg)
    assert value is not None and np.isfinite(value)
    assert abs(value - 0.5 * (parts["fcm_mlp"] + parts["xgb_maps"])) < 1e-9

    # Same public functions the bridge calls, on the same rows, including the gap flag at a cut history.
    mix = mix_bridge.mixmod
    hist_end = day_start.to_pydatetime()

    def records_of(frame):
        return [
            {
                "direction": row.direction,
                "bin_ts": row.bin_ts.to_pydatetime(),
                "dur_min": float(row.dur_min),
                "congestion_ratio": float(row.congestion_ratio),
                "speed_kmh": float(row.speed_kmh),
            }
            for row in frame.itertuples(index=False)
        ]

    def placed(frame):
        rows = records_of(frame)
        times = [row["bin_ts"] for row in rows]
        grid = mix.fcm.Grid.spanning(min(times), max(times))
        history_first, _ = mix.fcm.sgt_day_bounds(
            mix.fcm.parse_sgt_date(cfg.history_start), mix.fcm.parse_sgt_date(cfg.history_start)
        )
        cut = grid.start_unix > mix.fcm.unix(history_first)
        return mix.bins_from_rows(rows, grid, cut=cut), grid

    train_bins, train_grid = placed(history)
    train = mix.build_data(
        cfg, train_bins, np.arange(train_grid.n), label_cutoff=hist_end, layer_a={}, protect_labels=False
    )
    serve_day = mix.fcm.date_to_epoch_day(mix.fcm.sgt_date(hist_end))
    models, _, _ = mix.fit_day(cfg, train, serve_day)
    scored_bins, scored_grid = placed(pd.concat([history, latest], ignore_index=True))
    origin_dt = origin.to_pydatetime()
    index = scored_grid.index(mix.fcm.floor_bin(origin_dt))
    scored = mix.build_data(
        cfg, scored_bins, np.asarray([index]), label_cutoff=origin_dt, layer_a={}, protect_labels=False
    )
    pred = mix.predict_direction(models, scored["SG_TO_MY"], np.asarray([0]))
    np.testing.assert_allclose(value, float(pred["mix"][0]))
    np.testing.assert_allclose(parts["fcm_mlp"], float(pred["fcm_mlp"][0]))
    np.testing.assert_allclose(parts["xgb_maps"], float(pred["xgb_maps"][0]))
