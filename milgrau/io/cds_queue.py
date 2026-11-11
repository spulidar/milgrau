"""Best-effort observation of the public CDS live dashboard.

The public dashboard is not a documented machine-readable queue API: a failed,
changed or stale response must never be treated as a low queue. No credentials,
job creation, cancellation, or CDS submission occur here.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from urllib.request import Request, urlopen

CDS_LIVE_URL = "https://cds.climate.copernicus.eu/live"


@dataclass(frozen=True)
class CDSQueueSnapshot:
    queued_requests: int
    source: str = CDS_LIVE_URL


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.excluded = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.excluded += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.excluded:
            self.excluded -= 1

    def handle_data(self, data: str) -> None:
        if not self.excluded:
            self.text.append(data)


def parse_cds_queue(html: str) -> CDSQueueSnapshot:
    """Read the explicitly labelled global queue number, never infer a value."""
    parser = _VisibleText()
    parser.feed(html)
    visible = " ".join(parser.text).replace("\u202f", " ").replace("\xa0", " ")
    match = re.search(r"\bQueued requests\s+([\d][\d\s,]*)", visible, re.IGNORECASE)
    if match is None:
        raise ValueError("CDS live dashboard has no readable 'Queued requests' metric.")
    number = re.sub(r"\D", "", match.group(1))
    if not number:
        raise ValueError("CDS global queue metric is missing.")
    return CDSQueueSnapshot(queued_requests=int(number))


def read_cds_queue(*, timeout_seconds: float = 4.0) -> CDSQueueSnapshot:
    """Read one public dashboard snapshot with a short bounded network timeout."""
    request = Request(
        CDS_LIVE_URL,
        headers={"User-Agent": "MILGRAU/queue-observation", "Accept": "text/html"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        data = response.read(1_000_001)
    if len(data) > 1_000_000:
        raise ValueError("CDS live dashboard response exceeded maximum size.")
    return parse_cds_queue(data.decode("utf-8", errors="replace"))
