#!/usr/bin/env python3
"""
Phase 5 analysis: consume per_object.csv produced by run_pipeline_sample.py.
Generates historical_characterization.md, integrity_audit.csv,
false_alarm_characterization.md, and summary.md.
"""
import csv, json, sys, statistics as stats
from pathlib import Path
from datetime import date, datetime
from collections import Counter

OUT = Path('results/reentry/catalog_scale_validation')
PER_OBJECT = OUT / 'per_object.csv'
MANIFEST = Path('data/reentry/catalog_scale_tmp/sample_manifest.json')
ORIGINAL_AGG = Path('results/reentry/aggregate_summary.csv')
TEST_IDS_FILE = Path('artifacts/reentry/test_norad_ids.json')
GCAT_TSV = Path('data/reentry/gcat_cache/satcat.tsv')

LOW_PERI_THRESHOLD = 300.0   # km; periapsis at decay below this = physically plausible reentry


def load_per_object():
    with open(PER_OBJECT) as f:
        return list(csv.DictReader(f))


def _f(v):
    try: return float(v)
    except (ValueError, TypeError): return None


def load_manifest():
    return json.loads(MANIFEST.read_text())


def load_gcat():
    if not GCAT_TSV.exists(): return {}
    hdr = None; g = {}
    with open(GCAT_TSV) as f:
        for line in f:
            if line.startswith('#JCAT'):
                hdr = line.lstrip('#').rstrip('\n').split('\t'); continue
            if line.startswith('#') or not line.strip(): continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < len(hdr): continue
            rec = {hdr[i].strip(): fields[i].strip() for i in range(len(hdr))}
            nid = rec.get('Satcat','').strip()
            if nid: g[nid] = rec
    return g


GCAT_MONTHS = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,
               'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
import re
_GCAT_DATE_RE = re.compile(r'^(\d{4})\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?(\?)?\s*$')
def parse_gcat_date(s):
    if not s or s == '-': return None, False
    m = _GCAT_DATE_RE.match(s.strip())
    if not m: return None, False
    y, mon, d, hm, q = m.groups()
    try: return date(int(y), GCAT_MONTHS[mon], int(d)), bool(q)
    except Exception: return None, False


def parse_iso(s):
    if not s: return None
    return datetime.fromisoformat(s[:10]).date()


def get_peri_at_decay(nid, decay_date):
    """Periapsis at (nearest TLE to) decay_date. Reads from gp_history_cache."""
    cf = Path(f'data/reentry/gp_history_cache/{nid}.json')
    if not cf.exists() or not decay_date:
        return None, 0
    with open(cf) as f: recs = json.load(f)
    if not recs: return None, 0
    target = datetime.combine(decay_date, datetime.min.time())
    best, best_delta = None, None
    post_count = 0
    for r in recs:
        try: ep = datetime.fromisoformat(r['EPOCH'][:19])
        except Exception: continue
        if ep.date() > decay_date:
            post_count += 1
        delta = abs((ep - target).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta; best = r
    peri = None
    if best and 'PERIAPSIS' in best:
        try: peri = float(best['PERIAPSIS'])
        except Exception: pass
    return peri, post_count


def pct(xs, p):
    xs = sorted(xs)
    if not xs: return None
    k = (len(xs)-1) * p
    f, c = int(k), min(int(k)+1, len(xs)-1)
    return xs[f] + (xs[c]-xs[f])*(k-f)


def dist_stats(xs, label):
    xs = [x for x in xs if x is not None]
    if not xs: return f'{label}: n=0\n'
    return (f'{label}: n={len(xs)}  mean={stats.mean(xs):.4f}  '
            f'median={stats.median(xs):.4f}  '
            f'p25={pct(xs,0.25):.4f}  p75={pct(xs,0.75):.4f}  '
            f'p90={pct(xs,0.90):.4f}  p95={pct(xs,0.95):.4f}  '
            f'min={min(xs):.4f}  max={max(xs):.4f}\n')


def main():
    rows = load_per_object()
    manifest = load_manifest()
    print(f'loaded {len(rows)} per-object rows')

    for r in rows:
        r['max_p_failure_f'] = _f(r['max_p_failure'])
        r['decay_date_parsed'] = parse_iso(r['decay_date'])

    # Step 3: Historical reentry characterization
    # Filter: periapsis at decay < 300 km (physically plausible reentry)
    print('computing peri-at-decay for all objects...')
    plausible = []
    implausible = []
    for r in rows:
        peri, post = get_peri_at_decay(r['norad_id'], r['decay_date_parsed'])
        r['peri_at_decay'] = peri
        r['post_decay_tle_count'] = post
        if peri is None or peri >= LOW_PERI_THRESHOLD:
            implausible.append(r)
        else:
            plausible.append(r)
    print(f'plausible (peri<{LOW_PERI_THRESHOLD}): {len(plausible)}')
    print(f'implausible / unknown peri:              {len(implausible)}')

    # Distributions
    pl_maxp = [r['max_p_failure_f'] for r in plausible]
    all_maxp = [r['max_p_failure_f'] for r in rows]
    p_below_05 = sum(1 for v in pl_maxp if v is not None and v < 0.5)
    p_below_025 = sum(1 for v in pl_maxp if v is not None and v < 0.25)

    # Lead times at each threshold (only for plausible objects and those that crossed)
    leads = {th: [] for th in ['0.10','0.25','0.50','0.75']}
    for r in plausible:
        for th in leads:
            v = _f(r.get(f'lead_p{th.replace("0.","")}_days',''))
            # column names: lead_p10_days, lead_p25_days, lead_p50_days, lead_p75_days
            col = 'lead_p' + th.replace('0.','').rstrip('0') + '_days'
            col_normal = {'0.10':'lead_p10_days','0.25':'lead_p25_days',
                          '0.50':'lead_p50_days','0.75':'lead_p75_days'}[th]
            v = _f(r.get(col_normal,''))
            if v is not None: leads[th].append(v)

    # 78-object reference distribution
    if ORIGINAL_AGG.exists():
        orig_rows = list(csv.DictReader(open(ORIGINAL_AGG)))
        orig_maxp = [_f(r['P_forward_max']) for r in orig_rows]
        orig_leads = {th: [] for th in ['0.10','0.25','0.50','0.75']}
        for r in orig_rows:
            for th in orig_leads:
                col = 'lead_'+th
                v = _f(r.get(col,''))
                if v is not None: orig_leads[th].append(v)
    else:
        orig_maxp = []; orig_leads = {th:[] for th in ['0.10','0.25','0.50','0.75']}

    hist = []
    hist.append('# STTS-Reentry — Catalog-Scale Historical Characterization (sample)\n')
    hist.append(f'Sample size: **{len(rows)}** objects (of 3,623 targeted; '
                f'rows emitted by pipeline after feature-extraction validation).\n')
    hist.append(f'Periapsis-at-decay < {LOW_PERI_THRESHOLD:.0f} km (plausible natural decay): '
                f'**{len(plausible)}**.\n')
    hist.append(f'Periapsis-at-decay ≥ {LOW_PERI_THRESHOLD:.0f} km or unknown: **{len(implausible)}**.\n\n')
    hist.append('## max P(failure) distribution\n\n')
    hist.append('```\n')
    hist.append(dist_stats(pl_maxp, 'plausible-only'))
    hist.append(dist_stats(all_maxp, 'all objects  '))
    if orig_maxp:
        hist.append(dist_stats(orig_maxp, '78-object set'))
    hist.append('```\n\n')
    hist.append(f'Objects with max P < 0.5:  **{p_below_05}** of {len(plausible)} plausible '
                f'(below tactical threshold)\n')
    hist.append(f'Objects with max P < 0.25: **{p_below_025}** of {len(plausible)} plausible '
                f'(bimodal-outlier signature)\n\n')
    hist.append('## Lead-time distribution (days before decay_date at which P_forward first crossed threshold)\n\n')
    hist.append('```\n')
    for th,xs in leads.items():
        hist.append(dist_stats(xs, f'lead @P≥{th}  '))
    if orig_leads.get('0.50'):
        hist.append('\n78-object reference:\n')
        for th,xs in orig_leads.items():
            hist.append(dist_stats(xs, f'  lead @P≥{th}'))
    hist.append('```\n')
    (OUT / 'historical_characterization.md').write_text(''.join(hist))
    print(f'wrote historical_characterization.md')

    # Step 4: integrity audit
    print('loading GCAT...')
    gcat = load_gcat()
    print(f'GCAT entries: {len(gcat)}')

    candidates = [r for r in rows if r['max_p_failure_f'] is not None and r['max_p_failure_f'] < 0.25]
    integrity_rows = []
    for r in candidates:
        nid = r['norad_id']
        g = gcat.get(nid, {})
        gdate, gunc = parse_gcat_date(g.get('DDate',''))
        corpus_date = r['decay_date_parsed']
        offset = None
        if corpus_date and gdate:
            offset = (corpus_date - gdate).days
        classification = 'likely_framework_edge_case'
        if offset is not None and abs(offset) > 30:
            classification = 'likely_catalog_error'
        elif not gdate:
            classification = 'unclassified_no_gcat'
        integrity_rows.append({
            'norad_id': nid,
            'corpus_decay_date': r['decay_date'],
            'gcat_ddate': gdate.isoformat() if gdate else '',
            'gcat_status': g.get('Status',''),
            'gcat_uncertain': int(gunc),
            'date_offset_days': offset if offset is not None else '',
            'post_decay_tle_count': r.get('post_decay_tle_count',''),
            'max_p_failure': r['max_p_failure'],
            'classification': classification,
        })

    with open(OUT / 'integrity_audit.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'norad_id','corpus_decay_date','gcat_ddate','gcat_status',
            'gcat_uncertain','date_offset_days','post_decay_tle_count',
            'max_p_failure','classification'])
        w.writeheader()
        for rr in integrity_rows:
            w.writerow(rr)
    print(f'wrote integrity_audit.csv: {len(integrity_rows)} rows')

    # Step 5: false alarm characterization
    # "Valid natural decay": peri<300km at decay, GCAT DDate matches corpus within 7d, no post-decay TLE
    valid_natural = []
    for r in plausible:
        nid = r['norad_id']
        g = gcat.get(nid, {})
        gdate, _ = parse_gcat_date(g.get('DDate',''))
        cdate = r['decay_date_parsed']
        gstat = g.get('Status','').strip()
        if not gstat or not gdate or not cdate: continue
        if abs((cdate - gdate).days) > 7: continue
        if r.get('post_decay_tle_count', 0) > 0: continue
        if gstat not in ('R','F'): continue
        valid_natural.append(r)

    vn_maxp = [r['max_p_failure_f'] for r in valid_natural]
    fa_count_05 = sum(1 for v in vn_maxp if v is not None and v < 0.5)

    fa = []
    fa.append('# STTS-Reentry — False-Alarm Characterization (sample)\n\n')
    fa.append(f'Valid-natural-decay subset: **{len(valid_natural)}** objects.\n')
    fa.append(f'Definition: GCAT status R or F, |DDate − corpus decay| ≤ 7 d, '
              f'peri-at-decay < {LOW_PERI_THRESHOLD:.0f} km, no post-decay TLE.\n\n')
    fa.append(f'Objects with max P(failure) < 0.5 despite valid natural decay: '
              f'**{fa_count_05}** of {len(valid_natural)} '
              f'({100.0*fa_count_05/max(1,len(valid_natural)):.3f}%).\n\n')
    fa.append('## Distribution of max P(failure) over valid-natural subset\n\n')
    fa.append('```\n')
    fa.append(dist_stats(vn_maxp, 'valid-natural'))
    fa.append('```\n')
    (OUT / 'false_alarm_characterization.md').write_text(''.join(fa))
    print(f'wrote false_alarm_characterization.md')

    # Step 6: 78-object reproducibility
    test_ids = set(json.loads(TEST_IDS_FILE.read_text()))
    orig = {r['norad_id']: r for r in orig_rows} if orig_maxp else {}
    repro_rows = []
    for r in rows:
        if r['norad_id'] in test_ids and r['norad_id'] in orig:
            new = r['max_p_failure_f']
            old = _f(orig[r['norad_id']]['P_forward_max'])
            if old is None or new is None: continue
            repro_rows.append((r['norad_id'], old, new, abs(old-new)))
    exact_match = sum(1 for x in repro_rows if x[3] == 0.0)
    max_diff = max((x[3] for x in repro_rows), default=0.0)
    print(f'78-object reproducibility: {exact_match}/{len(repro_rows)} byte-identical  '
          f'max |Δ|={max_diff}')

    # Step 7: Summary
    sm = []
    sm.append('# STTS-Reentry — Catalog-Scale Validation Summary (sample, n≈3,623)\n\n')
    sm.append('## Section 1 — Population\n\n')
    sm.append(f'Full reentry-class population (Historical DECAY class, since 2018-01-01): '
              f'**19,615**.\nStratified sample: 500/year × 9 years, all years having ≥500 '
              f'sampled fully; years with <500 took all. Union with 78 original test IDs. '
              f'Total sample: **3,623 objects**. Ran to completion: **{len(rows)}**.\n\n')
    sm.append('## Section 2 — Historical characterization\n\n')
    sm.append(f'Plausible natural-decay subset (peri<{LOW_PERI_THRESHOLD:.0f} km at decay): '
              f'{len(plausible)}. Bimodal-outlier count (max P < 0.25): {p_below_025}. '
              f'Below tactical threshold (< 0.5): {p_below_05}.\n\n')
    sm.append('## Section 3 — Integrity audit\n\n')
    sm.append(f'{len(candidates)} objects flagged max P < 0.25. By GCAT cross-reference: '
              f'{sum(1 for r in integrity_rows if r["classification"]=="likely_catalog_error")} '
              f'likely-catalog-error, '
              f'{sum(1 for r in integrity_rows if r["classification"]=="likely_framework_edge_case")} '
              f'likely-framework-edge-case, '
              f'{sum(1 for r in integrity_rows if r["classification"]=="unclassified_no_gcat")} '
              f'unclassified.\n\n')
    sm.append('## Section 4 — False alarm\n\n')
    sm.append(f'Valid natural-decay subset: {len(valid_natural)}. '
              f'max P < 0.5 among valid: {fa_count_05} '
              f'({100.0*fa_count_05/max(1,len(valid_natural)):.3f}%).\n\n')
    sm.append('## Section 5 — 78-object reproducibility\n\n')
    sm.append(f'Of the 78 original test objects, {len(repro_rows)} present in sample run. '
              f'Byte-identical max P: **{exact_match}/{len(repro_rows)}**. '
              f'Max observed |Δ| = {max_diff}.\n\n')
    if exact_match == len(repro_rows) and len(repro_rows) == 78:
        sm.append('STATUS: PASS — catalog-scale run reproduces the 78-object result exactly.\n\n')
    else:
        sm.append('STATUS: FLAG — pipeline drift detected; downstream results must be reviewed.\n\n')
    sm.append('## Section 6 — Edge cases\n\n')
    if integrity_rows:
        sm.append('See integrity_audit.csv for the full list of framework-flagged '
                  'candidates. Per-classification counts above.\n')
    (OUT / 'summary.md').write_text(''.join(sm))
    print(f'wrote summary.md')

    # Also dump the reproducibility table
    with open(OUT / 'reproducibility_78.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['norad_id','orig_max_p','sample_max_p','abs_delta'])
        for row in sorted(repro_rows): w.writerow(row)
    print('wrote reproducibility_78.csv')

if __name__ == '__main__':
    main()
