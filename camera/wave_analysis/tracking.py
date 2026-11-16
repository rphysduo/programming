"""Human-reviewed temporal associations with actual timestamps and reuse guards."""
import numpy as np


def fit_track(observations, time_sigma_s=None):
    """Fit signed speed with >=3 distinct-time observations and report uncertainty.

    Regression residual error and conditional measurement error are separate.
    Missing exposure uncertainty leaves total speed uncertainty unavailable.
    """
    if len(observations)<3:
        return dict(speed_m_s=None, residual_speed_se_m_s=None, conditional_speed_sigma_m_s=None,
                    status='need_three_independent_observations')
    t = np.array([o['time_s'] for o in observations], float)
    y = np.array([o['y_m'] for o in observations], float)
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<=0):
        raise ValueError('Track times must strictly increase and positions must be finite')
    centered = t-t.mean()
    denom = centered@centered
    speed = float(centered@y/denom)
    residual = y-(y.mean()+speed*centered)
    residual_se = float(np.sqrt((residual@residual)/(len(t)-2)/denom))
    sigma = [o.get('sigma_y_m') for o in observations]
    total = None
    if time_sigma_s is not None and all(s is not None for s in sigma):
        if not np.isfinite(time_sigma_s) or time_sigma_s<0 or any(not np.isfinite(s) or s<0 for s in sigma):
            raise ValueError('Uncertainties must be finite and nonnegative')
        variance = np.asarray(sigma)**2+speed**2*time_sigma_s**2
        total = float(np.sqrt(np.sum((centered/denom)**2*variance)))
    return dict(speed_m_s=speed, residual_speed_se_m_s=residual_se,
                conditional_speed_sigma_m_s=total, fit_rmse_m=float(np.sqrt(np.mean(residual**2))),
                duration_s=float(t[-1]-t[0]), intervals_s=np.diff(t).tolist(), status='reviewed_linear_fit')


def reviewed_tracks(profiles, features, links, config):
    """Resolve explicit crest/packet identities, never nearest-neighbor guesses.

    A link is one uninterrupted track segment. Researchers split tracks at
    merging, splitting, ambiguous identity or breaking. Reused camera frames
    are omitted from a fit even when paired to a different frame in the other
    camera. Packet positions require explicit distinct-envelope evidence.
    """
    lookup = {p['pair_index']:p for p in profiles}
    crests = {(w['pair_index'],w['feature_id']):w for w in features}
    results, assigned, identifiers = [], set(), set()
    if not isinstance(links, list):
        raise ValueError('Track links must be a list')
    for link in links:
        if link.get('reviewed') is not True or not link.get('evidence', '').strip():
            raise ValueError('Track identity requires human review and evidence')
        identifier = link.get('track_id')
        if not isinstance(identifier,str) or not identifier or identifier in identifiers:
            raise ValueError('Track IDs must be unique strings')
        identifiers.add(identifier)
        kind = link.get('kind')
        if kind not in ('crest','packet'):
            raise ValueError('Track kind must be crest or packet')
        if kind=='packet' and link.get('distinct_envelope') is not True:
            raise ValueError('Group velocity requires a distinct observable envelope')
        accepted, rejected, used0, used1 = [], [], set(), set()
        items = link.get('observations', [])
        if not isinstance(items,list):
            raise ValueError('Track observations must be a list')
        for item in items:
            if not isinstance(item,dict) or item.get('pair_index') not in lookup:
                raise ValueError('Track references an unknown profile pair')
            if kind=='crest' and (item['pair_index'],item.get('feature_id')) not in crests:
                raise ValueError('Track references an unknown crest feature')
        seen_pairs = set()
        for item in sorted(items, key=lambda v: lookup[v['pair_index']]['time_s']):
            pi = item['pair_index']
            if pi in seen_pairs:
                raise ValueError('Duplicate pair in a track')
            seen_pairs.add(pi)
            profile = lookup[pi]
            if profile['pair']['host_gap_ms']>config['max_host_gap_ms']:
                rejected.append(dict(pair_index=pi,reason='host_gap_exceeds_configured_limit'))
                continue
            if kind=='crest':
                key = (pi,item['feature_id'])
                if key in assigned:
                    raise ValueError('A crest cannot belong to two tracks')
                assigned.add(key)
                feature = crests[key]
                y, sigma = feature['crest_y_m'], feature['crest_sigma_y_m']
            else:
                y, sigma = item.get('y_m'), item.get('sigma_y_m')
                if not isinstance(y,(int,float)) or isinstance(y,bool) or not item.get('evidence', '').strip() or not np.isfinite(y):
                    raise ValueError('Packet observation needs measured Y and evidence')
                if sigma is not None and (not isinstance(sigma,(int,float)) or not np.isfinite(sigma) or sigma<0):
                    raise ValueError('Packet position uncertainty must be finite and nonnegative')
                if not config['y_range_mm'][0]<=y*1000<=config['y_range_mm'][1]:
                    raise ValueError('Packet observation outside measured region')
            f0,f1 = profile['pair']['frame_indices']
            if f0 in used0 or f1 in used1:
                rejected.append(dict(pair_index=pi,reason='shared_camera_frame'))
                continue
            if accepted and profile['time_s']-accepted[-1]['time_s']>config['max_track_gap_s']:
                raise ValueError('Track crosses a timing gap; split and review separate tracks')
            used0.add(f0)
            used1.add(f1)
            accepted.append(dict(pair_index=pi, time_s=profile['time_s'], y_m=y, sigma_y_m=sigma,
                                 feature_id=item.get('feature_id'),
                                 frame_indices=[f0,f1], host_gap_ms=profile['pair']['host_gap_ms']))
        results.append(dict(track_id=identifier,kind=kind,evidence=link['evidence'],observations=accepted,
                            rejected=rejected,**fit_track(accepted,config['exposure_time_sigma_s'])))
    return results
