#!/usr/bin/env python3
"""
Catalog-scale pipeline driver. Runs the validated STTS-Reentry pipeline
against a sampled subset of the full reentry-class population.

Read-only against pipeline code. Constructs a minimal in-memory
satellites dict per object (reads gp_history_cache + injects decay_date
from the sample manifest), calls run_single_object from the aggregate
module, writes per-object and aggregate CSVs.
"""
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reentry.config import K_NEIGHBORS
from reentry.energy_state_aggregate import run_single_object
from reentry.train import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

SAMPLE_MANIFEST = Path('data/reentry/catalog_scale_tmp/sample_manifest.json')
CACHE_DIR = Path('data/reentry/gp_history_cache')

OUT_DIR = Path('results/reentry/catalog_scale_validation')
PER_OBJECT_DIR = OUT_DIR / 'per_object'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_OBJECT_DIR.mkdir(parents=True, exist_ok=True)
PER_OBJECT_CSV = OUT_DIR / 'per_object.csv'
DIAG_LOG = OUT_DIR / 'run_diagnostics.txt'


def build_sat_dict(nid, manifest_entry):
    cf = CACHE_DIR / f'{nid}.json'
    if not cf.exists():
        return None, 'no_cache_file'
    with open(cf) as f:
        records = json.load(f)
    if not records:
        return None, 'empty_cache'
    # Drop any records missing EPOCH
    records = [r for r in records if 'EPOCH' in r]
    # Object name / intldes
    name = ''
    intldes = ''
    for r in records:
        if r.get('OBJECT_NAME') and not name: name = r['OBJECT_NAME']
        if r.get('OBJECT_ID') and not intldes: intldes = r['OBJECT_ID']
        if name and intldes: break
    if not name:
        name = manifest_entry.get('OBJECT_NAME','')
    if not intldes:
        intldes = manifest_entry.get('OBJECT_ID','')
    decay_date = (manifest_entry.get('DECAY_EPOCH') or '')[:10]
    return ({
        'norad_id': nid,
        'object_name': name,
        'intldes': intldes,
        'decay_epoch': decay_date,
        'classification': 'reentry',
        'n_tle_records': len(records),
        'tle_records': [_record_for_pipeline(r) for r in records],
        'is_storm': False,
    }, None)


STATE_CHANNELS = ['PERIAPSIS','MEAN_MOTION','MEAN_MOTION_DOT','BSTAR','ECCENTRICITY','APOAPSIS']


def _record_for_pipeline(r):
    out = {'epoch': r.get('EPOCH')}
    for ch in STATE_CHANNELS:
        try:
            out[ch] = float(r[ch])
        except (KeyError, TypeError, ValueError):
            out[ch] = None
    return out


def main(limit=None, only_cached=False):
    manifest = json.loads(SAMPLE_MANIFEST.read_text())
    sample_ids = sorted(manifest.keys(), key=int)

    if only_cached:
        sample_ids = [nid for nid in sample_ids if (CACHE_DIR / f'{nid}.json').exists()]
        log.info(f'limited to {len(sample_ids)} cached IDs')

    if limit:
        sample_ids = sample_ids[:limit]

    log.info(f'loading model + Markov artifacts...')
    model = load_model()
    scaler, W, lda = model['scaler'], model['W'], model['lda']
    kmeans = model['kmeans']
    markov = model['markov']
    transition_matrix = markov['transition_matrix']
    failure_cells = markov['failure_cells']
    failure_arr = np.array(sorted(set(failure_cells.tolist())))
    mu_nominal = float(markov['mu_nominal'].item())
    expected_entropy = markov['expected_entropy']
    train_proj = markov['train_proj']
    y_tr = markov['train_labels']
    cum_trans = np.cumsum(transition_matrix, axis=1)
    k = K_NEIGHBORS

    rng = np.random.RandomState(42)

    summaries = []
    diagnostics = []

    for i, nid in enumerate(sample_ids):
        sat_dict, err = build_sat_dict(nid, manifest[nid])
        if sat_dict is None:
            diagnostics.append((nid, 'skip', err))
            continue
        satellites = {nid: sat_dict}
        try:
            result = run_single_object(
                nid, satellites, scaler, W, lda, kmeans,
                transition_matrix, cum_trans, failure_arr,
                mu_nominal, expected_entropy, train_proj, y_tr,
                k, rng,
            )
        except Exception as e:
            diagnostics.append((nid, 'error', str(e)))
            continue
        if result is None:
            diagnostics.append((nid, 'skip', 'no windows after feature extraction'))
            continue
        # Move per-object CSV to our own directory (written by run_single_object
        # under results/reentry/per_object/). We leave it alongside.
        # Annotate the summary with diag fields.
        result['n_tle_total'] = len(sat_dict['tle_records'])
        summaries.append(result)
        if (i+1) % 25 == 0 or i == len(sample_ids)-1:
            log.info(f'  [{i+1}/{len(sample_ids)}] {nid} '
                     f'n_win={result["n_windows"]}  maxP={result["P_forward_max"]:.4f}')

    # Write per-object CSV in our output dir
    fieldnames = [
        'norad_id','decay_date','total_tles','n_windows',
        'max_p_failure','lead_p10_days','lead_p25_days','lead_p50_days','lead_p75_days',
        'cell_support_exits','artifact_flag_count','total_transitions','gate_exits',
        'object_name',
    ]
    with open(PER_OBJECT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow({
                'norad_id': s['norad_id'],
                'decay_date': s.get('decay_epoch',''),
                'total_tles': s.get('n_tle_total',''),
                'n_windows': s['n_windows'],
                'max_p_failure': round(s['P_forward_max'], 6),
                'lead_p10_days': s['lead_0.10'] if s['lead_0.10'] is not None else '',
                'lead_p25_days': s['lead_0.25'] if s['lead_0.25'] is not None else '',
                'lead_p50_days': s['lead_0.50'] if s['lead_0.50'] is not None else '',
                'lead_p75_days': s['lead_0.75'] if s['lead_0.75'] is not None else '',
                'cell_support_exits': 0,       # not currently wired; placeholder
                'artifact_flag_count': s.get('artifact_count', 0),
                'total_transitions': max(0, s['n_windows'] - 1),
                'gate_exits': 0,               # not currently wired; placeholder
                'object_name': s.get('object_name',''),
            })

    with open(DIAG_LOG, 'w') as f:
        f.write('norad_id\tstatus\treason\n')
        for nid, st, reason in diagnostics:
            f.write(f'{nid}\t{st}\t{reason}\n')

    log.info(f'wrote {len(summaries)} rows to {PER_OBJECT_CSV}')
    log.info(f'diagnostics: {len(diagnostics)} entries in {DIAG_LOG}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--only-cached', action='store_true')
    args = ap.parse_args()
    main(limit=args.limit, only_cached=args.only_cached)
