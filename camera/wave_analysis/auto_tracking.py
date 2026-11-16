"""Conservative temporal proposals; identity remains subject to human review."""
from collections import Counter
import numpy as np
from .tracking import fit_track


def packet_candidates(features, contrast=0.25):
    """Locate an interior crest-height envelope peak with resolved lower flanks.

    This is a sparse envelope proxy, not a Hilbert envelope or a second phase
    speed. Uniform trains, boundary peaks and multiple peaks are unavailable.
    """
    result = []
    groups = {}
    for f in features:
        groups.setdefault((f['pair_index'], f['segment']), []).append(f)
    for (pair, segment), values in groups.items():
        values = sorted(values, key=lambda f: f['crest_y_m'])
        if len(values) < 5:
            continue
        h = np.array([f['height_m'] for f in values])
        peaks = [i for i in range(1, len(h)-1) if h[i] > h[i-1] and h[i] > h[i+1]]
        if len(peaks) != 1:
            continue
        i = peaks[0]
        if i < 2 or i > len(h)-3 or min(h[i]-h[0], h[i]-h[-1]) < contrast*h[i]:
            continue
        f = values[i]
        spacing = max(f['crest_y_m']-values[i-1]['crest_y_m'], values[i+1]['crest_y_m']-f['crest_y_m'])
        result.append(dict(f, feature_id=f'packet_{segment}', crest_sigma_y_m=spacing/2))
    return result


def propose_tracks(profiles, features, config, *, max_speed_m_s, position_gate_m=0.01,
                   shape_fraction=0.5, kind='crest', breaking_times=()):
    """Link only unique motion-and-shape-compatible observations.

    A many-to-one/one-to-many candidate graph ends identities rather than
    resolving a merge/split arbitrarily. Small gaps are bridged and flagged;
    long gaps, explicit breaking boundaries and reused frames stop continuity.
    Speeds on proposals are provisional, never accepted dataset measurements.
    """
    if kind not in ('crest', 'packet') or any(not np.isfinite(v) or v <= 0 for v in
                                            (max_speed_m_s, position_gate_m, shape_fraction)):
        raise ValueError('Supply crest/packet and positive finite tracking gates')
    if any(not np.isfinite(t) for t in breaking_times):
        raise ValueError('Breaking times must be finite')
    times = [p['time_s'] for p in profiles]
    if not np.isfinite(times).all() or len(set(times)) != len(times):
        raise ValueError('Profiles need distinct finite timestamps')
    lookup = {p['pair_index']: p for p in profiles}
    if len(lookup) != len(profiles) or any(f['pair_index'] not in lookup for f in features):
        raise ValueError('Features must reference unique profile pairs')
    candidates = packet_candidates(features) if kind == 'packet' else features
    tracks, active, diagnostics, used0, used1 = [], [], [], set(), set()
    for step, p in enumerate(sorted(profiles, key=lambda p: p['time_s'])):
        time, pair = p['time_s'], p['pair_index']
        f0, f1 = p['pair']['frame_indices']
        reason = ('shared_camera_frame' if f0 in used0 or f1 in used1 else
                  'host_gap' if p['pair']['host_gap_ms'] > config['max_host_gap_ms'] else None)
        if reason:
            diagnostics.append(dict(pair_index=pair, reason=reason))
            continue
        used0.add(f0); used1.add(f1)
        surviving = []
        for track in active:
            last = track['observations'][-1]
            reason = ('timing_gap' if time-last['time_s'] > config['max_track_gap_s'] else
                      'breaking_boundary' if any(last['time_s'] < t <= time for t in breaking_times) else None)
            if reason:
                track['flags'].append(reason)
            else:
                surviving.append(track)
        active = surviving
        fs = [f for f in candidates if f['pair_index'] == pair]
        graph = {}
        for ti, track in enumerate(active):
            obs = track['observations']; last = obs[-1]; dt = time-last['time_s']
            prediction = last['y_m']
            if len(obs) >= 2:
                prediction += (last['y_m']-obs[-2]['y_m'])/(last['time_s']-obs[-2]['time_s'])*dt
            gate = position_gate_m + (max_speed_m_s*dt if len(obs) < 2 else 0)
            graph[ti] = [fi for fi, f in enumerate(fs)
                if abs(f['crest_y_m']-last['y_m']) <= max_speed_m_s*dt+position_gate_m
                and abs(f['crest_y_m']-prediction) <= gate
                and abs(f['height_m']-last['height_m']) <= shape_fraction*max(last['height_m'], 1e-12)
                and (f.get('wavelength_m') is None or last.get('wavelength_m') is None or
                     abs(f['wavelength_m']-last['wavelength_m']) <= shape_fraction*last['wavelength_m'])]
        degree = Counter(fi for neighbors in graph.values() for fi in neighbors)
        matched, next_active = set(), []
        for ti, track in enumerate(active):
            neighbors = graph[ti]
            if len(neighbors) == 1 and degree[neighbors[0]] == 1:
                fi = neighbors[0]; matched.add(fi)
                if step-track['_step'] > 1:
                    track['flags'].append('bridged_missing_observation')
                track['observations'].append(_observation(fs[fi], p)); track['_step'] = step
                next_active.append(track)
            elif neighbors:
                track['flags'].append('ambiguous_merge_split_or_identity')
                diagnostics.append(dict(pair_index=pair, track_id=track['track_id'], reason=track['flags'][-1]))
            else:
                next_active.append(track)  # may return within the configured gap
        for fi, f in enumerate(fs):
            if fi in matched:
                continue
            track = dict(track_id=f'auto_{kind}_{len(tracks):04d}', kind=kind, reviewed=False,
                         evidence='', observations=[_observation(f, p)], flags=[], _step=step)
            if kind == 'packet':
                track['distinct_envelope'] = False  # reviewer must verify observability
            tracks.append(track); next_active.append(track)
        active = next_active
    for track in tracks:
        track.pop('_step')
        track['flags'] = sorted(set(track['flags']))
        track['provisional_fit'] = fit_track(track['observations'], config['exposure_time_sigma_s'])
        track['provisional_fit']['status'] = 'machine_identity_unreviewed'
    return dict(tracks=tracks, diagnostics=diagnostics, kind=kind,
                settings=dict(max_speed_m_s=max_speed_m_s, position_gate_m=position_gate_m,
                              shape_fraction=shape_fraction),
                note='Review identities and split ambiguous tracks before measure --tracks. Packet positions are sparse height-envelope proxies.')


def _observation(f, profile):
    return dict(pair_index=f['pair_index'], feature_id=f['feature_id'], time_s=profile['time_s'],
                y_m=f['crest_y_m'], sigma_y_m=f.get('crest_sigma_y_m'), height_m=f['height_m'],
                wavelength_m=f.get('wavelength_m'), evidence='', frame_indices=profile['pair']['frame_indices'])
