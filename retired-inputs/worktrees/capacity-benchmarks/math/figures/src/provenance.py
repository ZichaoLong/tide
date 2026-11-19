"""Record definitions, source files and rendering inputs for a reviewed batch."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess

import cairocffi
import cairosvg
import PIL

from drawing import FONT_DIR, MATH_FONT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_local_provenance(base, output_dir, registry):
    base, output_dir = Path(base), Path(output_dir)
    fonts = [FONT_DIR/'NotoSansCJK-Regular.ttc', FONT_DIR/'NotoSansCJK-Bold.ttc', MATH_FONT]
    generators = [base/'render.py', base/'requirements.txt', *sorted((base/'src').glob('*.py'))]
    result = {
        'schema_version': 1,
        'definitions_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=base, text=True).strip(),
        'rendering': {
            'python': platform.python_version(), 'Pillow': PIL.__version__,
            'CairoSVG': cairosvg.__version__, 'cairo': cairocffi.cairo_version_string(),
            'canvas': [2400, 1350],
            'fonts': [{'path': str(path), 'sha256': sha256(path),
                       'face_index': 2 if path.suffix == '.ttc' else 0} for path in fonts],
        },
        'generators': {str(path.relative_to(base)): sha256(path) for path in generators},
        'figures': {},
    }
    for name, spec in registry.items():
        result['figures'][name] = {
            'title': spec['title'], 'purpose': spec['purpose'],
            'assumptions': spec['assumptions'],
            'generator': str(Path(inspect.getsourcefile(spec['draw'])).relative_to(base)),
            'function': spec['draw'].__name__,
            'definitions': [{'reference': ref, 'sha256': sha256(base/ref.split('#')[0])}
                            for ref in spec['sources']],
            'artifacts': {name+ext: sha256(output_dir/(name+ext)) for ext in ('.svg', '.png')},
        }
    # Keep the reviewed historical snapshot separate from routine render runs.
    destination = output_dir/'local-render-checks.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return destination
