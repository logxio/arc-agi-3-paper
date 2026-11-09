"""Count Writeup words the way a reader would: body, tables and captions count; the
References section, HTML comments, image paths and link URLs do not."""
import re
import sys
from pathlib import Path

text = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "paper" / "frontier-lock.md").read_text()
text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
text = re.split(r"^## References\s*$", text, flags=re.M)[0]
text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)          # images
text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)         # links -> text
text = re.sub(r"^\|?[\s:|-]+\|?\s*$", " ", text, flags=re.M)  # table rules
words = [w for w in re.split(r"[\s|]+", text) if re.search(r"[A-Za-z0-9]", w)]
print(len(words))
