#!/usr/bin/env python3
"""Package validated sweep scores without host paths or training configuration."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.summary.read_text())
    if raw.get('state') != 'complete' or not raw.get('all_candidates_evaluated'):
        parser.error('Require a complete checkpoint sweep.')
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in raw['ranked_results']:
        rows.append({key: item[key] for key in ('iteration', 'metrics', 'episode_count', 'success_count')})
    rows.sort(key=lambda row: (row['success_count'], row['metrics']['spl'], row['iteration']), reverse=True)
    result = {'split': 'R2R-CE val-unseen', 'episode_count': 1839,
              'checkpoint_range': {'start': 12000, 'end': 20000, 'interval': 200},
              'evaluated_checkpoints': len(rows), 'seed': 0,
              'selection': ['SR', 'SPL', 'iteration'], 'best': rows[0], 'ranked_results': rows}
    with (destination / 'r2rce_val_unseen.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    with (destination / 'ranking.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'iteration', 'episodes', 'successes', 'SR_percent', 'SPL_percent', 'nDTW_percent', 'NE_m'])
        for index, row in enumerate(rows, 1):
            m = row['metrics']
            writer.writerow([index, row['iteration'], row['episode_count'], row['success_count'],
                             m['success']*100, m['spl']*100, m['ndtw']*100, m['distance_to_goal']])
    print('Packaged {} checkpoint results; best iteration {}'.format(len(rows), rows[0]['iteration']))


if __name__ == '__main__':
    main()
