"""Mirror the legacy Pillow primitives to editable SVG, keeping PNG rendering intact."""
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import ImageDraw


class VectorDrawing:
    """The small ImageDraw subset used by the three restored figure scripts."""

    def __init__(self, image, *, scale, background):
        self._draw = ImageDraw.Draw(image)
        self.scale = scale
        self._serial = 0
        width, height = (size / scale for size in image.size)
        self.root = ET.Element('svg', {
            'xmlns': 'http://www.w3.org/2000/svg',
            'width': f'{width:g}', 'height': f'{height:g}',
            'viewBox': f'0 0 {width:g} {height:g}',
            'version': '1.1', 'xml:lang': 'zh-CN',
        })
        self._add('rect', {
            'x': '0', 'y': '0', 'width': f'{width:g}', 'height': f'{height:g}',
            'fill': background,
        })

    def _n(self, value):
        return f'{value / self.scale:g}'

    def _add(self, tag, attributes):
        self._serial += 1
        return ET.SubElement(self.root, tag, {
            'id': f'{tag}-{self._serial:03d}', **attributes,
        })

    def _paint(self, fill, outline=None, width=1):
        return {
            'fill': fill or 'none', 'stroke': outline or 'none',
            'stroke-width': self._n(width),
        }

    def textlength(self, text, font):
        return self._draw.textlength(text, font=font)

    def text(self, xy, text, *, font, fill, anchor='mm'):
        # Use the original font metrics to translate Pillow's vertical anchor
        # to an SVG alphabetic baseline. Keep whole strings as editable text.
        if anchor not in ('mm', 'lm', 'rm'):
            raise ValueError(f'Unsupported legacy text anchor: {anchor}')
        self._draw.text(xy, text, font=font, fill=fill, anchor=anchor)
        anchored = font.getbbox(text, anchor=anchor)
        baseline = font.getbbox(text, anchor='ls')
        family, style = font.getname()
        element = self._add('text', {
            'x': self._n(xy[0]),
            'y': self._n(xy[1] + anchored[1] - baseline[1]),
            'font-family': f'{family}, sans-serif',
            'font-size': self._n(font.size),
            'font-weight': '700' if 'Bold' in style else '400',
            'text-anchor': {'m': 'middle', 'l': 'start', 'r': 'end'}[anchor[0]],
            'fill': fill, 'xml:space': 'preserve', 'style': 'white-space: pre',
        })
        element.text = text

    def rounded_rectangle(self, xy, radius, *, fill, outline=None, width=1):
        self._draw.rounded_rectangle(xy, radius=radius, fill=fill,
                                     outline=outline, width=width)
        # Pillow draws outlines inside their bounds; SVG centers the stroke.
        inset = width / 2 if outline else 0
        left, top, right, bottom = xy
        self._add('rect', {
            'x': self._n(left + inset), 'y': self._n(top + inset),
            'width': self._n(right - left - 2 * inset),
            'height': self._n(bottom - top - 2 * inset),
            'rx': self._n(max(0, radius - inset)),
            **self._paint(fill, outline, width),
        })

    def ellipse(self, xy, *, fill, outline=None, width=1):
        self._draw.ellipse(xy, fill=fill, outline=outline, width=width)
        inset = width / 2 if outline else 0
        left, top, right, bottom = xy
        self._add('ellipse', {
            'cx': self._n((left + right) / 2),
            'cy': self._n((top + bottom) / 2),
            'rx': self._n((right - left) / 2 - inset),
            'ry': self._n((bottom - top) / 2 - inset),
            **self._paint(fill, outline, width),
        })

    def line(self, xy, *, fill, width=1, joint=None):
        self._draw.line(xy, fill=fill, width=width, joint=joint)
        self._add('polyline', {
            'points': ' '.join(f'{self._n(x)},{self._n(y)}' for x, y in xy),
            'fill': 'none', 'stroke': fill, 'stroke-width': self._n(width),
            'stroke-linejoin': 'round' if joint == 'curve' else 'miter',
        })

    def polygon(self, xy, *, fill):
        self._draw.polygon(xy, fill=fill)
        previous = self.root[-1]
        element = self._add('polygon', {
            'points': ' '.join(f'{self._n(x)},{self._n(y)}' for x, y in xy),
            'fill': fill,
        })
        # Each legacy arrow is a line immediately followed by its arrowhead.
        # Group those objects so a graphical editor can move the whole arrow.
        if previous.tag == 'polyline':
            self.root.remove(previous)
            self.root.remove(element)
            group = self._add('g', {})
            group.extend((previous, element))

    def save_svg(self, path, *, source):
        path = Path(path)
        title = ET.Element('title')
        title.text = path.stem
        description = ET.Element('desc')
        description.text = (
            f'由 src/legacy/{source} 的原始绘图坐标导出；'
            '文字、形状与连线可编辑。字体为 Noto Sans CJK SC。'
            '数学定义与来源见同目录 README.md 和 PROVENANCE.md。'
        )
        self.root.insert(0, description)
        self.root.insert(0, title)
        ET.indent(self.root, space='  ')
        ET.ElementTree(self.root).write(path, encoding='utf-8', xml_declaration=True)
        return path
