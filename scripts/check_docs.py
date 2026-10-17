#!/usr/bin/env python3
"""Validate a built LaFA documentation site without fetching external URLs.

Run after ``mkdocs build --strict``. Metadata checks cover sitemap pages;
local links and JSON-LD are checked in every generated HTML file except 404.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit


SITE_URL = "https://izzatum.github.io/The-Langevin-algorithms-Family/"
REPOSITORY = "https://github.com/izzatum/The-Langevin-algorithms-Family"
MIT_LICENSES = {
    "https://opensource.org/license/mit",
    "https://opensource.org/licenses/MIT",
    REPOSITORY + "/blob/main/LICENSE",
}


def normalized(value: str) -> str:
    return " ".join(value.split())


class Page(HTMLParser):
    """Collect the HTML contracts used by the validator, preserving line numbers."""

    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.titles: list[list[str]] = []
        self.headings: list[list[str]] = []
        self.meta: dict[str, list[str]] = defaultdict(list)
        self.canonicals: list[str] = []
        self.anchors: set[str] = set()
        self.references: list[tuple[int, str, str]] = []
        self.json_ld: list[str] = []
        self.base_urls: list[str] = []
        self._in_title = False
        self._in_h1 = False
        self._json_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if attrs.get("id"):
            self.anchors.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.anchors.add(attrs["name"])
        if tag == "title":
            self.titles.append([])
            self._in_title = True
        elif tag == "h1":
            self.headings.append([])
            self._in_h1 = True
        elif tag == "meta":
            key = attrs.get("name") or attrs.get("property")
            if key:
                self.meta[key.lower()].append(attrs.get("content") or "")
        elif tag == "link" and "canonical" in (attrs.get("rel") or "").lower().split():
            self.canonicals.append(attrs.get("href") or "")
        elif tag == "base":
            self.base_urls.append(attrs.get("href") or "")
        elif tag == "script":
            if (attrs.get("type") or "").lower().split(";", 1)[0].strip() == "application/ld+json":
                self._json_parts = []
        for attribute in ("href", "src"):
            value = attrs.get(attribute)
            if value is not None and tag != "base":
                self.references.append((self.getpos()[0], f"{tag}[{attribute}]", value))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False
        elif tag == "script" and self._json_parts is not None:
            self.json_ld.append("".join(self._json_parts))
            self._json_parts = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.titles[-1].append(data)
        if self._in_h1:
            self.headings[-1].append(data)
        if self._json_parts is not None:
            self._json_parts.append(data)


def json_objects(value: object):
    """Walk both individual JSON-LD documents and @graph/list documents."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_objects(child)


class Validator:
    def __init__(self, site_dir: Path, site_url: str) -> None:
        self.site_dir = site_dir.resolve()
        self.site_url = site_url
        self.origin = urlsplit(site_url)
        self.base_path = unquote(self.origin.path)
        self.pages: dict[Path, Page] = {}
        self.errors: list[str] = []
        self.checked_links = 0
        self.checked_json = 0
        self.sitemap_urls: list[str] = []

    def error(self, source: str | Path, message: str) -> None:
        if isinstance(source, Path):
            source = str(source.relative_to(self.site_dir))
        self.errors.append(f"{source}: {message}")

    def local_file(self, url: str) -> Path:
        """Map a public project URL to its file, including directory indexes."""
        parsed = urlsplit(url)
        path = unquote(parsed.path)
        if not path.startswith(self.base_path):
            raise ValueError(f"path escapes site base {self.base_path!r}: {path!r}")
        relative = path[len(self.base_path):]
        candidate = (self.site_dir / relative).resolve()
        if not candidate.is_relative_to(self.site_dir):
            raise ValueError(f"path escapes generated site: {path!r}")
        if candidate.is_dir() or path.endswith("/"):
            candidate = candidate / "index.html"
        return candidate

    def page_url(self, path: Path) -> str:
        relative = path.relative_to(self.site_dir).as_posix()
        if relative == "index.html":
            relative = ""
        elif relative.endswith("/index.html"):
            relative = relative[:-len("index.html")]
        return urljoin(self.site_url, quote(relative, safe="/"))

    def read_pages(self) -> None:
        for path in sorted(self.site_dir.rglob("*.html")):
            # MkDocs 404 links are resolved from the requested URL by its runtime.
            if path == self.site_dir / "404.html":
                continue
            page = Page(path)
            try:
                page.feed(path.read_text(encoding="utf-8"))
                page.close()
            except (OSError, UnicodeError, ValueError) as exc:
                self.error(path, f"cannot parse HTML: {exc}")
                continue
            self.pages[path] = page
        if not self.pages:
            self.error("site", "no generated HTML pages found; run mkdocs build --strict first")

    def read_sitemap(self) -> dict[Path, str]:
        sitemap = self.site_dir / "sitemap.xml"
        try:
            root = ET.parse(sitemap).getroot()
        except (OSError, ET.ParseError) as exc:
            self.error(sitemap, f"cannot read sitemap: {exc}")
            return {}
        if root.tag.rsplit("}", 1)[-1] != "urlset":
            self.error(sitemap, "expected a sitemap urlset")
        self.sitemap_urls = [node.text.strip() if node.text else ""
                             for node in root.iter()
                             if node.tag.rsplit("}", 1)[-1] == "loc"]
        if not self.sitemap_urls:
            self.error(sitemap, "sitemap contains no page URLs")
        if self.site_url not in self.sitemap_urls:
            self.error(sitemap, f"homepage URL missing: {self.site_url}")
        pages: dict[Path, str] = {}
        seen: set[str] = set()
        for url in self.sitemap_urls:
            if url in seen:
                self.error(sitemap, f"duplicate URL: {url}")
            seen.add(url)
            try:
                parsed = urlsplit(url)
                if (parsed.scheme, parsed.netloc) != (self.origin.scheme, self.origin.netloc):
                    raise ValueError("URL must be absolute and use the configured site origin")
                if parsed.query or parsed.fragment:
                    raise ValueError("sitemap URL must not contain a query or fragment")
                path = self.local_file(url)
            except ValueError as exc:
                self.error(sitemap, f"{url!r}: {exc}")
                continue
            if path == self.site_dir / "404.html":
                self.error(sitemap, "404.html must not be indexed")
            elif path not in self.pages:
                self.error(sitemap, f"{url}: no generated HTML file at {path.relative_to(self.site_dir)}")
            elif path in pages:
                self.error(sitemap, f"multiple URLs resolve to {path.relative_to(self.site_dir)}")
            else:
                pages[path] = url
        return pages

    def one_value(self, page: Page, label: str, values: list[str]) -> str:
        if len(values) != 1 or not normalized(values[0]):
            self.error(page.path, f"expected exactly one nonempty {label}; found {len(values)}")
            return ""
        return normalized(values[0])

    def check_metadata(self, sitemap_pages: dict[Path, str]) -> None:
        titles: dict[str, Path] = {}
        descriptions: dict[str, Path] = {}
        for path, url in sitemap_pages.items():
            page = self.pages[path]
            self.one_value(page, "H1", ["".join(parts) for parts in page.headings])
            title = self.one_value(page, "HTML title", ["".join(parts) for parts in page.titles])
            description = self.one_value(page, "meta description", page.meta["description"])
            for label, value, seen in (("title", title, titles),
                                       ("description", description, descriptions)):
                if value in seen:
                    self.error(path, f"duplicate {label} also used by {seen[value].relative_to(self.site_dir)}")
                elif value:
                    seen[value] = path
            canonical = self.one_value(page, "canonical URL", page.canonicals)
            if canonical != url:
                self.error(path, f"canonical must equal sitemap URL {url!r}; got {canonical!r}")
            for key, expected in (("og:title", title), ("og:description", description),
                                  ("og:url", url), ("twitter:title", title),
                                  ("twitter:description", description)):
                actual = self.one_value(page, key, page.meta[key])
                if actual != expected:
                    self.error(path, f"{key} must match page metadata; expected {expected!r}, got {actual!r}")
            card = self.one_value(page, "twitter:card", page.meta["twitter:card"])
            if card not in {"summary", "summary_large_image"}:
                self.error(path, f"unsupported Twitter card {card!r}")
            for directive in page.meta["robots"] + page.meta["googlebot"]:
                if {"noindex", "none"} & set(directive.lower().replace(",", " ").split()):
                    self.error(path, "sitemap page has a noindex robots directive")

    def check_json_ld(self) -> None:
        homepage_objects: list[dict] = []
        forbidden = {"aggregaterating", "review", "reviews", "ratingvalue", "ratingcount", "reviewcount"}
        for path, page in self.pages.items():
            for index, raw in enumerate(page.json_ld, start=1):
                self.checked_json += 1
                try:
                    document = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self.error(path, f"JSON-LD block {index} is invalid JSON: {exc}")
                    continue
                if not isinstance(document, (dict, list)):
                    self.error(path, f"JSON-LD block {index} must contain an object or array")
                for obj in json_objects(document):
                    types = obj.get("@type", [])
                    if isinstance(types, str):
                        types = [types]
                    if not isinstance(types, list):
                        self.error(path, f"JSON-LD block {index} has invalid @type")
                        types = []
                    if forbidden & {key.lower() for key in obj} or any(
                        item in ("Review", "Rating", "AggregateRating") for item in types
                    ):
                        self.error(path, "JSON-LD must not contain unsupported ratings or reviews")
                    if path == self.site_dir / "index.html" and "SoftwareSourceCode" in types:
                        homepage_objects.append(obj)
        if len(homepage_objects) != 1:
            self.error("index.html", f"expected one SoftwareSourceCode JSON-LD object; found {len(homepage_objects)}")
            return
        software = homepage_objects[0]
        if software.get("codeRepository") != REPOSITORY:
            self.error("index.html", f"SoftwareSourceCode.codeRepository must be {REPOSITORY}")
        license_url = software.get("license")
        if not isinstance(license_url, str) or license_url.rstrip("/") not in MIT_LICENSES:
            self.error("index.html", "SoftwareSourceCode.license must link to the MIT license")
        version_fields = [software[key] for key in ("version", "softwareVersion") if key in software]
        if not version_fields or any(value != "0.1.0" for value in version_fields):
            self.error("index.html", "SoftwareSourceCode version must be 0.1.0")

    def check_links(self) -> None:
        for path, page in self.pages.items():
            base_url = self.page_url(path)
            if len(page.base_urls) > 1:
                self.error(path, "multiple base elements make link resolution ambiguous")
            if page.base_urls:
                base_url = urljoin(base_url, page.base_urls[0])
            for line, kind, value in page.references:
                value = value.strip()
                if not value or value.startswith("//"):
                    continue
                source = f"{path.relative_to(self.site_dir)}:{line} {kind}={value!r}"
                try:
                    destination = urlsplit(urljoin(base_url, value))
                    if destination.scheme not in {"http", "https"}:
                        continue
                    if destination.netloc.lower() != self.origin.netloc.lower():
                        continue
                    self.checked_links += 1
                    target = self.local_file(destination.geturl())
                except ValueError as exc:
                    self.error(source, str(exc))
                    continue
                if not target.is_file():
                    self.error(source, f"missing file {target.relative_to(self.site_dir)}")
                    continue
                # Browser text-fragment directives do not refer to HTML element IDs.
                fragment = unquote(destination.fragment).split(":~:", 1)[0]
                if fragment and target in self.pages and fragment not in self.pages[target].anchors:
                    # HTML defines #top as a valid document-top destination.
                    if fragment.lower() != "top":
                        self.error(source, f"missing fragment #{fragment} in {target.relative_to(self.site_dir)}")

    def check_citations(self) -> None:
        references_path = self.local_file(urljoin(self.site_url, "references/"))
        page = self.pages.get(references_path)
        if page is None:
            self.error("references/index.html", "bibliography page is missing")
            return
        for number in range(1, 25):
            anchor = f"r{number:02d}"
            if anchor not in page.anchors:
                self.error(references_path, f"missing stable citation anchor #{anchor}")

    def run(self) -> int:
        self.read_pages()
        sitemap_pages = self.read_sitemap()
        self.check_metadata(sitemap_pages)
        self.check_json_ld()
        self.check_links()
        self.check_citations()
        errors = list(dict.fromkeys(self.errors))
        for error in errors:
            print(f"ERROR {error}", file=sys.stderr)
        status = "FAILED" if errors else "PASSED"
        print(f"Documentation validation {status}: {len(sitemap_pages)} sitemap pages, "
              f"{len(self.pages)} HTML files, {self.checked_links} internal links/assets, "
              f"{self.checked_json} JSON-LD blocks, {len(errors)} errors.")
        return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=Path("site"),
                        help="MkDocs output directory (default: site)")
    parser.add_argument("--site-url", default=SITE_URL,
                        help=f"absolute public URL with trailing slash (default: {SITE_URL})")
    args = parser.parse_args()
    try:
        parsed = urlsplit(args.site_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or parsed.query or parsed.fragment or not parsed.path.endswith("/")):
            raise ValueError("expected an absolute HTTP(S) URL with a trailing slash and no query or fragment")
    except ValueError as exc:
        parser.error(f"invalid --site-url: {exc}")
    return Validator(args.site_dir, args.site_url).run()


if __name__ == "__main__":
    raise SystemExit(main())
