"""Geometric wave measurements on supported continuous profile segments.

Babanin et al. (2007), Fig. 1 and equations 1–2, DOI 10.1029/2006GL029135:
Sk=a1/a2-1; As=b_front/b_rear-1 using mean-level crossings around a crest.
For unequal neighboring troughs this implementation explicitly uses their mean
depth. This is a convention for irregular profiles, not a claim from the paper.
"""
import numpy as np


def linear_reference(wavelength_m, depth_m, g=9.81):
    """Finite-depth linear theory reference speeds, separate from observations."""
    if not all(np.isfinite(v) and v > 0 for v in (wavelength_m, depth_m, g)):
        raise ValueError('Wavelength, depth and gravity must be finite and positive')
    k = 2*np.pi/wavelength_m
    kh = k*depth_m
    cp = np.sqrt(g*np.tanh(kh)/k)
    term = 0.0 if kh > 350 else 2*kh/np.sinh(2*kh)
    return dict(cp_linear_m_s=float(cp), cg_linear_m_s=float(cp*(1+term)/2), kh=float(kh))


def extrema(values, maximum=True):
    """Return internal extrema, using the midpoint of a flat plateau."""
    v = np.asarray(values) * (1 if maximum else -1)
    result, i = [], 1
    while i < len(v)-1:
        end = i
        while end+1 < len(v) and v[end+1] == v[i]:
            end += 1
        if end < len(v)-1 and v[i] > v[i-1] and v[end] > v[end+1]:
            result.append((i+end)//2)
        i = end+1
    return result


def smooth(y, eta, window):
    """Optional local-linear smoothing on actual positions; never crosses gaps."""
    if window == 1:
        return eta.copy()
    out = eta.copy()
    half = window//2
    for i in range(half, len(y)-half):
        sl = slice(i-half, i+half+1)
        out[i] = np.polyfit(y[sl]-y[i], eta[sl], 1)[1]
    return out


def crossing(y0, y1, e0, e1):
    """Linear still-water crossing between bracketing measured samples."""
    return float(y0-e0*(y1-y0)/(e1-e0))


def measure_segment(points, config, segment_id=0):
    """Return local slopes and identifiable crests, retaining unavailable values.

    Crests require two interior troughs and the configured minimum samples and
    height. Wavelength is distance to the next eligible crest along propagation.
    Derivatives describe the sampled/smoothed profile, not unresolved curvature.
    """
    if len(points) < 3:
        return [], []
    y = np.asarray([p['y_m'] for p in points], float)
    raw = np.asarray([p['eta_m'] for p in points], float)
    if not np.isfinite(y).all() or not np.isfinite(raw).all() or np.any(np.diff(y)<=0):
        raise ValueError('Profiles need finite elevation and strictly increasing Y')
    eta = smooth(y, raw, config['smoothing_window'])
    slope = np.gradient(eta, y, edge_order=2)
    slope_rows = [dict(point_id=p['point_id'], y_m=float(y[i]), eta_raw_m=float(raw[i]),
        eta_used_m=float(eta[i]), slope=float(slope[i]), slope_sigma=None,
        derivative_method='nonuniform three-point finite difference', segment=segment_id)
        for i,p in enumerate(points)]
    if config['smoothing_window']==1 and all(p.get('sigma_eta_m') is not None for p in points):
        for i,row in enumerate(slope_rows):
            indices = [0,1,2] if i==0 else ([len(y)-3,len(y)-2,len(y)-1] if i==len(y)-1 else [i-1,i,i+1])
            dx=y[indices]-y[i]
            # Polynomial-exact three-point derivative weights on uneven spacing.
            weights=np.linalg.solve(np.array([np.ones(3),dx,dx**2]),[0,1,0])
            row['slope_sigma']=float(np.sqrt(sum(w*w*points[j]['sigma_eta_m']**2 for w,j in zip(weights,indices))))
    troughs = extrema(eta, maximum=False)
    candidates = extrema(eta)
    waves = []
    direction = config['propagation_direction']
    for i in candidates:
        left = [j for j in troughs if j<i]
        right = [j for j in troughs if j>i]
        if not left or not right:
            continue
        l, r = left[-1], right[0]
        if r-l+1 < config['min_samples_per_wave']:
            continue
        # Mean over the complete trough-to-trough interval, without changing XYZ
        # or the exported eta relative to the measured still-water reference.
        reference = float(np.sum((eta[l:r]+eta[l+1:r+1])*.5*np.diff(y[l:r+1]))/(y[r]-y[l])) if config['shape_reference']=='local_wave_mean' else 0.0
        centered = eta-reference
        a1, a2 = centered[i], -(centered[l]+centered[r])/2
        height = a1+a2
        if height*1000 < config['min_wave_height_mm'] or min(a1,a2)<=0:
            continue
        lo = [j for j in range(l,i) if centered[j]<=0 and centered[j+1]>0]
        hi = [j for j in range(i,r) if centered[j]>=0 and centered[j+1]<0]
        bfront = brear = asymmetry = None
        if lo and hi:
            j,k = lo[-1],hi[0]
            back_cross = crossing(y[j],y[j+1],centered[j],centered[j+1])
            front_cross = crossing(y[k],y[k+1],centered[k],centered[k+1])
            distances = (front_cross-y[i], y[i]-back_cross)
            bfront, brear = distances if direction==1 else distances[::-1]
            asymmetry = bfront/brear-1 if min(bfront,brear)>0 else None
        depth = config['still_water_depth_mm']
        bottom = config['bottom_profile_yz_mm']
        if bottom is not None:
            bottom = np.asarray(bottom)
            depth = config['still_water_z_mm']-np.interp(y[i]*1000,bottom[:,0],bottom[:,1]) if bottom[0,0]<=y[i]*1000<=bottom[-1,0] else None
        elif config['ramp_start_y_mm'] is not None and y[i]*1000 >= config['ramp_start_y_mm']:
            depth = None  # Constant depth is not extended over an unmeasured ramp.
        depth = depth/1000 if depth is not None and depth>0 else None
        front = slope[i:r+1] if direction==1 else slope[l:i+1]
        rear = slope[l:i+1] if direction==1 else slope[i:r+1]
        sigma = [points[j].get('sigma_eta_m') for j in (l,i,r)]
        height_sigma = None if any(s is None for s in sigma) or config['smoothing_window']!=1 else float(np.sqrt(sigma[1]**2+(sigma[0]**2+sigma[2]**2)/4))
        sampling_halfwidth=float(max(y[i]-y[i-1],y[i+1]-y[i])/2)
        position_sigma=points[i].get('sigma_y_m')
        crest_sigma=None if position_sigma is None else float(np.sqrt(position_sigma**2+sampling_halfwidth**2/3))
        waves.append(dict(feature_id=f's{segment_id}_p{i}', segment=segment_id, crest_point_id=points[i]['point_id'],
            crest_y_m=float(y[i]), crest_eta_m=float(eta[i]), trough_left_y_m=float(y[l]), trough_right_y_m=float(y[r]),
            crest_amplitude_m=float(a1), trough_depth_m=float(a2), shape_reference_m=reference,
            shape_reference_kind=config['shape_reference'],
            height_m=float(height), height_sigma_m=height_sigma, wavelength_m=None, wavelength_sigma_m=None,
            steepness=None, steepness_sigma=None, skewness_sigma=None, asymmetry_sigma=None,
            skewness=float(a1/a2-1), asymmetry=None if asymmetry is None else float(asymmetry),
            front_width_m=bfront, rear_width_m=brear,
            max_front_slope=float(np.max(np.abs(front))), max_rear_slope=float(np.max(np.abs(rear))),
            depth_m=depth, H_over_h=None if depth is None else float(height/depth), kh=None,
            cp_linear_m_s=None, cg_linear_m_s=None, crest_sigma_y_m=crest_sigma,
            crest_sampling_halfwidth_m=sampling_halfwidth,
            smoothing_height_change_m=float(height-(raw[i]-(raw[l]+raw[r])/2)),
            sample_count=r-l+1, flags=['conditional_measurement', 'extrema_limited_by_sampling']))
    ordered = sorted(waves, key=lambda w: direction*w['crest_y_m'])
    for a,b in zip(ordered,ordered[1:]):
        low,high=sorted((a['crest_y_m'],b['crest_y_m']))
        if any(low < y[i] < high for i in candidates):
            a['flags'].append('unresolved_intervening_crest')
            continue
        wavelength = abs(b['crest_y_m']-a['crest_y_m'])
        a.update(wavelength_m=wavelength, steepness=float(np.pi*a['height_m']/wavelength))
        if a['crest_sigma_y_m'] is not None and b['crest_sigma_y_m'] is not None:
            a['wavelength_sigma_m']=float(np.hypot(a['crest_sigma_y_m'],b['crest_sigma_y_m']))
            if a['height_sigma_m'] is not None:
                a['steepness_sigma']=float(a['steepness']*np.hypot(a['height_sigma_m']/a['height_m'],a['wavelength_sigma_m']/wavelength))
        if a['depth_m'] is not None:
            a.update(linear_reference(wavelength,a['depth_m']))
    for wave in waves:
        if wave['wavelength_m'] is None:
            wave['flags'].append('no_next_supported_crest')
        if wave['asymmetry'] is None:
            wave['flags'].append('missing_reference_level_crossings')
    return waves, slope_rows


def measure_profiles(profiles, config):
    """Measure each gap-separated segment independently with original provenance."""
    features, slopes, quality = [], [], []
    for profile in profiles:
        count = 0
        for segment in sorted({p['segment'] for p in profile['points']}):
            points = [p for p in profile['points'] if p['segment']==segment]
            waves, derivatives = measure_segment(points, config, segment)
            common = dict(pair_index=profile['pair_index'], time_s=profile['time_s'],
                          coverage_fraction=profile['coverage_fraction'])
            features.extend(dict(w, **common) for w in waves)
            slopes.extend(dict(s, **common) for s in derivatives)
            count += len(waves)
        quality.append(dict(pair_index=profile['pair_index'], supported_waves=count,
                            status='measured' if count else 'insufficient_wave_structure'))
    return features, slopes, quality
