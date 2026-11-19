#!/usr/bin/env python3
"""Render ignored Tide artifacts from tracked source files."""
import argparse
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['all', 'legacy', 'local'], default='all')
    parser.add_argument('--output-dir', type=Path, default=BASE)
    parser.add_argument('--only', nargs='+', help='Local figure names to render.')
    parser.add_argument('--record-provenance', action='store_true',
                        help='Write local-render-checks.json for the complete local batch.')
    args = parser.parse_args()
    if args.group == 'legacy' and (args.only or args.record_provenance):
        parser.error('--only and --record-provenance require group local or all')
    if args.only and args.record_provenance:
        parser.error('record provenance after rendering the complete local group')
    if args.group in ('all', 'local'):
        sys.path.insert(0, str(BASE / 'src'))
        from local_figures import REGISTRY
        names = args.only or list(REGISTRY)
        unknown = set(names) - REGISTRY.keys()
        if unknown:
            parser.error('unknown local figures: '+', '.join(sorted(unknown)))
    if args.group in ('all', 'legacy') and not args.only:
        for name in ['tide_structure.py', 'psuf_simple.py', 'psuf_complete.py']:
            subprocess.run([sys.executable, str(BASE / 'src' / 'legacy' / name),
                            '--output-dir', str(args.output_dir.resolve())], check=True)
    if args.group in ('all', 'local'):
        for name in names:
            figure = REGISTRY[name]['draw']()
            print(figure.save(name, args.output_dir))
        if args.record_provenance:
            from provenance import write_local_provenance
            print(write_local_provenance(BASE, args.output_dir, REGISTRY))


if __name__ == '__main__':
    main()
