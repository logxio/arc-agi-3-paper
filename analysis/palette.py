"""The paper's one role palette on a white page. No red and no green anywhere: cobalt navy is the certificate-gated
planner and everything it keeps, ochre amber is Frontier Lock, open-loop probes, truncation and clock inflation,
slate indigo a third series, cool gray people, frames and rules. Five tiers per family, deep to pale.
"""
from matplotlib.colors import to_hex, to_rgb


def mix(a, b, t):
    a, b = to_rgb(a), to_rgb(b)
    return to_hex([x + (y - x) * t for x, y in zip(a, b)])


TIERS = {
    'navy': ['#0F2F46', '#1B4F72', '#2E6F9E', '#6299C4', '#C5DCEF'],
    'amber': ['#7A3E06', '#B85D10', '#D9822B', '#F0B46E', '#FBE5C8'],
    'indigo': ['#2F3248', '#4A4E69', '#7C83BC', '#B6BADB', '#E2E3F1'],
    'gray': ['#3B4450', '#6E7781', '#8C959F', '#D0D7DE', '#EEF2F7'],
}
PAGE = '#FFFFFF'
PANEL = '#F6F8FA'
FRAME = '#D0D7DE'

# per family: (stroke, mid, fill, wash)
HUES = {
    'navy': ('#1B4F72', '#6299C4', '#C5DCEF', mix('#C5DCEF', PAGE, .55)),
    'amber': ('#B85D10', '#F0B46E', '#FBE5C8', mix('#FBE5C8', PAGE, .55)),
    'indigo': ('#4A4E69', '#7C83BC', '#B6BADB', '#E2E3F1'),
    'gray': ('#6E7781', '#8C959F', '#D0D7DE', '#EEF2F7'),
}
ROLE = {'ours': 'navy', 'lock': 'amber', 'third': 'indigo', 'base': 'gray', 'focus': 'navy'}
INK = '#1F2328'
MUTED = '#57606A'
RULE = GRID = '#EEF2F7'


def hue(role):
    return ROLE.get(role, role)


def dark(role):
    return HUES[hue(role)][0]


def mid(role):
    return HUES[hue(role)][1]


def light(role):
    return HUES[hue(role)][2]


def register_fonts():
    """Register the bold instance when the installed text face is variable."""
    from pathlib import Path
    from tempfile import gettempdir
    from matplotlib import font_manager
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    path = font_manager.findfont('STIX Two Text')
    font = TTFont(str(path))
    if 'fvar' not in font:
        return
    target = Path(gettempdir()) / 'frontier-lock-STIXTwoText-Bold.ttf'
    if not target.exists():
        bold = instantiateVariableFont(font, {'wght': 700}, inplace=False)
        for record in bold['name'].names:
            if record.nameID in (2, 17):
                record.string = 'Bold'.encode(record.getEncoding())
        bold.save(target)
    font_manager.fontManager.addfont(target)
