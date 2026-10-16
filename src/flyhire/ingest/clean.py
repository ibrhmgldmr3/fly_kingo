"""İlan metni temizliği ve tekrar ayıklama için hash."""

import hashlib
import html
import re
from html.parser import HTMLParser

from flyhire.extract.schema import normalize_text

_BLOCK_TAGS = {"p", "div", "section", "br", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(fragment: str, escaped: bool = False) -> str:
    """İlan HTML'ini düz metne çevirir; paragrafları ve madde işaretlerini korur.

    Greenhouse içeriği HTML-escape edilmiş gelir (`&lt;p&gt;`), `escaped=True` önce onu çözer.
    """
    if escaped:
        fragment = html.unescape(fragment)
    parser = _TextExtractor()
    parser.feed(fragment)
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # <li><p>metin</p></li>: madde işaretini hemen alttaki satırla birleştir (boş <li> boş satırla
    # ayrıldığı için birleşmez, bir sonraki adımda silinir)
    text = re.sub(r"^- *\n(?=[^\n-])", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^- *$\n?", "", text, flags=re.MULTILINE)  # boş madde
    # ardışık maddeler arasındaki boş satırları kaldır
    text = re.sub(r"(^- [^\n]*)\n\n+(?=- )", r"\1\n", text, flags=re.MULTILINE)
    text = re.sub(r"(^- [^\n]*)\n\n+(?=- )", r"\1\n", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def content_hash(company: str, title: str, text: str) -> str:
    """Başlık + şirket + metnin hash'i. Aynı ilan farklı kaynaktan gelse de aynı değeri verir."""
    key = "\x1f".join(normalize_text(part) for part in (company, title, text))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
