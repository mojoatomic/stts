#!/usr/bin/env python3
"""
Parse bulk TLE zips for a specific NORAD ID set. One-off driver for
catalog-scale validation. Reuses parse_bulk_tles.process_zip.
Read-only against pipeline code.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reentry.parse_bulk_tles import process_zip, BULK_DIR, CACHE_DIR

TARGET_FILE = Path('data/reentry/catalog_scale_tmp/sample_missing_ids.txt')
SAMPLE_MANIFEST = Path('data/reentry/catalog_scale_tmp/sample_manifest.json')


def main():
    target_ids = set(TARGET_FILE.read_text().splitlines()) - {''}
    print(f'target IDs: {len(target_ids):,}')
    manifest = json.loads(SAMPLE_MANIFEST.read_text())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(BULK_DIR.glob('tle20*.txt.zip'))
    print(f'zip files: {[z.name for z in zip_files]}')

    satellites_touched = set()
    total_records = 0

    for zip_path in zip_files:
        print(f'\nprocessing {zip_path.name}...', flush=True)
        by_norad = process_zip(zip_path, target_ids)
        for nid, records in by_norad.items():
            cf = CACHE_DIR / f'{nid}.json'
            existing = []
            if cf.exists():
                with open(cf) as f: existing = json.load(f)
            existing_epochs = {r['EPOCH'] for r in existing if 'EPOCH' in r}
            new_recs = [r for r in records if r.get('EPOCH') not in existing_epochs]
            if new_recs:
                combined = existing + new_recs
                combined = [r for r in combined if 'EPOCH' in r]
                combined.sort(key=lambda r: r['EPOCH'])
                # Inject DECAY_DATE and OBJECT_NAME from manifest onto the first record
                if nid in manifest:
                    m = manifest[nid]
                    dd = (m.get('DECAY_EPOCH') or '')[:10]
                    name = m.get('OBJECT_NAME','')
                    if dd and combined:
                        combined[0] = {**combined[0], 'DECAY_DATE': dd, 'OBJECT_NAME': name}
                with open(cf, 'w') as f:
                    json.dump(combined, f)
            satellites_touched.add(nid)
            total_records += len(new_recs)
        print(f'  zip satellites: {len(by_norad):,}   new recs total: {total_records:,}', flush=True)

    print(f'\ntouched {len(satellites_touched)} satellites')

    # Final check: which target IDs still missing?
    cached = set(p.stem for p in CACHE_DIR.glob('*.json'))
    still_missing = target_ids - cached
    Path('data/reentry/catalog_scale_tmp/still_missing_after_bulk.txt').write_text(
        '\n'.join(sorted(still_missing, key=int)))
    print(f'after bulk: still_missing = {len(still_missing)}')


if __name__ == '__main__':
    main()
