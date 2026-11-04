"""Deterministic text perturbations used in the manuscript robustness table."""

import argparse
import csv
import json
import re
from pathlib import Path

COLORS = {'red': 'blue', 'blue': 'red', 'green': 'yellow', 'yellow': 'blue', 'black': 'white', 'white': 'black', 'gray': 'brown', 'grey': 'brown', 'orange': 'yellow', 'brown': 'gray', 'purple': 'green', 'pink': 'green'}

COLOR_RE = re.compile('\\b(?:' + '|'.join(COLORS) + ')\\b', re.I)

DETAIL_RULES = [('fire hydrants?', 'traffic bollard', 'traffic bollards'), ('traffic lights?', 'streetlamp', 'streetlamps'), ('traffic signals?', 'streetlamp', 'streetlamps'), ('traffic signs?', 'streetlamp', 'streetlamps'), ('road signs?', 'streetlamp', 'streetlamps'), ('signposts?', 'streetlamp', 'streetlamps'), ('streetlights?', 'traffic signal', 'traffic signals'), ('street lamps?', 'traffic signal', 'traffic signals'), ('lampposts?', 'signpost', 'signposts'), ('lamp posts?', 'signpost', 'signposts'), ('bollards?', 'fire hydrant', 'fire hydrants'), ('traffic cones?', 'bollard', 'bollards'), ('cones?', 'bollard', 'bollards'), ('crosswalks?', 'speed bump', 'speed bumps'), ('pedestrian crossings?', 'speed bump', 'speed bumps'), ('drainage grates?', 'manhole cover', 'manhole covers'), ('grates?', 'manhole cover', 'manhole covers'), ('manhole covers?', 'drainage grate', 'drainage grates'), ('bike racks?', 'bench', 'benches'), ('bicycle racks?', 'bench', 'benches'), ('railings?', 'fence', 'fences'), ('fences?', 'railing', 'railings'), ('barriers?', 'bollard', 'bollards'), ('bridges?', 'tunnel', 'tunnels'), ('overpasses?', 'underpass', 'underpasses'), ('bus stops?', 'kiosk', 'kiosks'), ('poles?', 'bollard', 'bollards'), ('hedges?', 'fence', 'fences'), ('shrubs?', 'lamp post', 'lamp posts'), ('trees?', 'streetlamp', 'streetlamps'), ('sidewalks?', 'cycle lane', 'cycle lanes'), ('lane markings?', 'rail track', 'rail tracks'), ('directional arrows?', 'painted circle', 'painted circles'), ('curbs?', 'guardrail', 'guardrails'), ('walls?', 'fence', 'fences'), ('buildings?', 'parking garage', 'parking garages'), ('signs?', 'streetlamp', 'streetlamps')]

DETAIL_RULES = [(re.compile('\\b' + pattern + '\\b', re.I), one, many) for pattern, one, many in DETAIL_RULES]

def preserve_case(old, new):
    return new.upper() if old.isupper() else new.capitalize() if old.istitle() else new

def remove_final_clause(text):
    separator = ';' if ';' in text else ','
    head, tail = text.rsplit(separator, 1)
    head = re.sub('\\b(?:and|while|with|whereas)\\s*$', '', head).rstrip(' ,;.')
    assert head and tail.strip()
    return (head + '.', tail.strip())

def wrong_color(text, landmark_edit=None):
    match = COLOR_RE.search(text)
    if match:
        old = match.group()
        new = preserve_case(old, COLORS[old.lower()])
        return (text[:match.start()] + new + text[match.end():], f'{old} -> {new}')
    assert landmark_edit is not None, 'A color-free caption needs a grounded landmark edit'
    target, replacement = (landmark_edit['target'], landmark_edit['replacement'])
    assert text.count(target) == 1, (target, text)
    return (text.replace(target, replacement, 1), f"{target} -> {replacement} ({landmark_edit['visual_answer']})")

def wrong_detail(text):
    for pattern, singular, plural in DETAIL_RULES:
        match = pattern.search(text)
        if match:
            old = match.group()
            is_plural = old.lower().endswith('s') and (not old.lower().endswith('ss'))
            new = preserve_case(old, plural if is_plural else singular)
            return (text[:match.start()] + new + text[match.end():], f'{old} -> {new}')
    return (text, '')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root / 'outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/qwen3_query_prompts_qwen-vl_77clip10.csv')
    parser.add_argument('--landmark-edits', type=Path, required=True, help='Local CSV of grounded replacements for color-free captions')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with args.source.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    with args.landmark_edits.open(newline='', encoding='utf-8-sig') as handle:
        edits = {r['image_name']: r for r in csv.DictReader(handle)}
    if not rows or len({r['image_name'] for r in rows}) != len(rows):
        raise ValueError('Expected unique source queries')
    output = {label: [] for label in ('controlled_incomplete', 'controlled_color', 'controlled_detail')}
    audit = []
    for row in rows:
        name, source = row['image_name'], row['description']
        incomplete, removed = remove_final_clause(source)
        color, color_edit = wrong_color(source, edits.get(name))
        detail, detail_edit = wrong_detail(source)
        if not color_edit or not detail_edit:
            raise ValueError(f'Missing perturbation for {name}')
        for label, text in zip(output, (incomplete, color, detail)):
            output[label].append({'image_name': name, 'description': text})
        audit.append({'image_name': name, 'removed_clause': removed,
                      'color_edit': color_edit, 'detail_edit': detail_edit})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for label, data in {**output, 'edits': audit}.items():
        with (args.output_dir / f'{label}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f'Generated three paired perturbations for {len(rows)} queries')


if __name__ == '__main__':
    main()
