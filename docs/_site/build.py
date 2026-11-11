#!/usr/bin/env python3
"""Generate transient Hugo content from the authoritative docs/ files."""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "getting-started": ["getting_started.md", "operating_milgrau.md", "configuration_reference.md", "terminology_units.md", "processing_levels.md"],
    "processing/level-0": ["libids.md", "level0_schema.md", "solar_regimes.md", "level0_validation_workbook.md"],
    "processing/level-1": ["lipancora.md", "level1_schema.md", "atmospheric_profiles.md", "level1_validation_workbook.md"],
    "processing/level-2": ["lebear.md", "level2_schema.md", "lidar_ratio_climatology_provenance.md"],
    "scientific-validation": ["scientific_quality.md", "scientific_traceability.md", "overlap_model.md", "spu_raman_channel_provenance.md"],
    "tools": ["explorer.md", "scc_batch_operator_guide.md", "scc_level0_interoperability.md"],
    "reference": ["netcdf_metadata_policy.md", "publication_integration.md", "code_inventory.md", "cleanup_review.md", "scc_native_client.md", "site_hugo.md"],
}
SECTIONS = {
    "getting-started": ("Getting started", "Install and operate MILGRAU", "rocket_launch", 10),
    "processing/level-0": ("Level 0 · LIBIDS", "Licel acquisition, sessions and QA", "database", 20),
    "processing/level-1": ("Level 1 · LIPANCORA", "Corrected signals and molecular atmosphere", "query_stats", 30),
    "processing/level-2": ("Level 2 · LEBEAR", "Aerosol optical retrieval", "show_chart", 40),
    "scientific-validation": ("Scientific validation", "Methods, evidence and limitations", "science", 50),
    "tools": ("Scientific tools", "Explorer and SCC interoperability", "build", 60),
    "reference": ("Reference", "Data products and provenance", "menu_book", 70),
    "history": ("Development history", "Archived engineering plans", "history", 90),
}
GROUP_OF = {name: group for group, items in GROUPS.items() for name in items}
LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
MARKER = ".milgrau-generated-docs"
GITHUB = "https://github.com/spulidar/milgrau/blob/main/"


def target(relative: Path) -> Path:
    if relative == Path("README.md"):
        return Path("docs/_index.md")
    if relative.parts[0] == "history":
        return Path("docs/history") / relative.name
    return Path("docs") / GROUP_OF.get(relative.name, "reference") / relative.name


def meta(title: str, desc: str, icon: str, weight: int) -> str:
    values = dict(title=title, description=desc, icon=icon, weight=weight, toc=True, katex=True, draft=False)
    out = ["---"]
    for key, value in values.items():
        rendered = ("true" if value else "false") if isinstance(value, bool) else str(value) if isinstance(value, int) else json.dumps(value, ensure_ascii=False)
        out.append(f"{key}: {rendered}")
    return "\n".join(out) + "\n---\n\n"


def links(body: str, relative: Path, routes: dict[Path, Path], root: Path) -> str:
    docs_dir = (root / "docs").resolve()

    def rewrite(match: re.Match[str]) -> str:
        label, href = match.groups()
        if href.startswith(("https:", "http:", "mailto:", "#", "/", "data:")):
            return match.group(0)
        path, sep, fragment = href.partition("#")
        if not path or " " in path:
            return match.group(0)
        source = (docs_dir / relative.parent / unquote(path)).resolve()
        if not source.is_file():
            raise ValueError(f"Missing linked source: {relative} -> {href}")
        try:
            rel_source = source.relative_to(docs_dir)
        except ValueError:
            try:
                repo_file = source.relative_to(root.resolve())
            except ValueError as err:
                raise ValueError(f"Link outside repository: {relative} -> {href}") from err
            return f"[{label}]({GITHUB}{repo_file.as_posix()}{sep}{fragment})"
        if rel_source in routes:
            route = routes[rel_source].as_posix()
            href = '{{< relref "' + route + '" >}}' + (sep + fragment if sep else "")
            return '<a href="' + href + '">' + html.escape(label) + '</a>'
        return match.group(0)

    output, fenced = [], False
    for line in body.splitlines(keepends=True):
        if line.lstrip().startswith(("~~~", chr(96) * 3)):
            fenced = not fenced
        output.append(line if fenced else LINK_RE.sub(rewrite, line))
    return "".join(output)


def build(root: Path = ROOT, output: Path | None = None) -> int:
    root = root.resolve()
    docs = root / "docs"
    dest = output.resolve() if output else root / "docs" / "_site" / ".hugo-content"
    if not docs.is_dir():
        raise FileNotFoundError(f"No docs source: {docs}")
    # Allow only the transient build folder inside docs/_site; all other
    # documentation and site infrastructure is protected.
    allowed_site_output = docs / "_site" / ".hugo-content"
    if dest in (root, docs) or (docs in dest.parents and dest != allowed_site_output):
        raise ValueError("Refusing to overwrite source documentation.")
    if dest.exists():
        if not (dest / MARKER).is_file():
            raise ValueError(f"Refusing unmanaged generated directory: {dest}")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    (dest / MARKER).write_text("Generated from docs/; never edit.\n", encoding="utf-8")
    # Ignore website build tooling and any previously generated Hugo Markdown.
    files = sorted(
        x for x in docs.rglob("*.md")
        if x.is_file() and x.relative_to(docs).parts[0] != "_site"
    )
    routes = {p.relative_to(docs): target(p.relative_to(docs)) for p in files}
    if len(set(routes.values())) != len(routes):
        raise ValueError("Duplicate site destinations")
    for group, (title, description, icon, weight) in SECTIONS.items():
        where = dest / "docs" / group / "_index.md"
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(meta(title, description, icon, weight) + description + "\n", encoding="utf-8")
    for source in files:
        rel = source.relative_to(docs)
        raw = source.read_text(encoding="utf-8")
        match = TITLE_RE.search(raw)
        title = match.group(1) if match else rel.stem.replace("_", " ").title()
        if match:
            raw = raw[:match.start()] + raw[match.end():]
        if rel == Path("README.md"):
            group, weight, title = "reference", 1, "MILGRAU documentation"
        elif rel.parts[0] == "history":
            group = "history"
            weight = SECTIONS[group][3] * 100 + 10 * (1 + files.index(source))
        else:
            group = GROUP_OF.get(rel.name, "reference")
            # Globally distinct weights matter: Lotus Docs' default next/prev
            # navigation can otherwise jump between sections sharing weight 10.
            # Each themed group has a stable block; pages follow GROUPS order.
            base = SECTIONS[group][3] * 100
            in_group = GROUPS.get(group, [])
            position = in_group.index(rel.name) + 1 if rel.name in in_group else 90 + files.index(source)
            weight = base + 10 * position
        _, description, icon, _ = SECTIONS[group]
        where = dest / routes[rel]
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(meta(title, description, icon, weight) + links(raw.lstrip("\n"), rel, routes, root), encoding="utf-8")
    # The project's GitHub README is also the editorial source for the
    # homepage overview. Site navigation/visuals live only in docs/_site/.
    # We intentionally omit README's Mermaid code fence here: the website
    # already draws the same scientific flow as accessible HTML.
    root_readme = root / "README.md"
    if root_readme.is_file():
        readme = root_readme.read_text(encoding="utf-8")
        def extract_section(heading: str) -> str:
            tag = "## " + heading + "\n"
            if readme.count(tag) != 1:
                raise ValueError(f"README lacks exactly one section: {heading}")
            block = readme.split(tag, 1)[1]
            return re.split(r"(?m)^## ", block, maxsplit=1)[0].strip()

        introduction = readme.split("## Scientific processing workflow", 1)[0]
        introduction = introduction.removeprefix("# MILGRAU\n").strip()
        workflow = extract_section("Scientific processing workflow")
        workflow = workflow.split("### Processing pathway at a glance", 1)[0]
        # Preserve only the README's explanatory lead and level overview
        # table; the detailed L0/L1/L2 methods belong in docs/.
        workflow = re.split(r"(?m)^### Level 0", workflow, maxsplit=1)[0].strip()
        installation = extract_section("Installation")
        scientific_quality = extract_section("Scientific quality and interpretation")
        homepage = (
            meta("MILGRAU", "Open scientific processing for atmospheric lidar", "science", 0)
            + "## About MILGRAU\n\n" + introduction
            + "\n\n## Scientific processing workflow\n\n" + workflow
            + "\n\n## Install MILGRAU\n\n" + installation
            + "\n\n## Scientific quality and interpretation\n\n" + scientific_quality
            + "\n"
        )
        # Links in the repo-root README point to docs/*.md; convert them
        # only in the temporary homepage copy, preserving GitHub Markdown.
        def home_link(match: re.Match[str]) -> str:
            label, href = match.groups()
            if not href.startswith("docs/"):
                return match.group(0)
            path, separator, fragment = href.partition("#")
            route = routes.get(Path(path).relative_to("docs"))
            if route is None:
                raise ValueError(f"Missing homepage documentation target: {href}")
            ref = '{{< relref "' + route.as_posix() + '" >}}'
            return '<a href="' + ref + (separator + fragment if separator else "") + '">' + html.escape(label) + '</a>'
        homepage = LINK_RE.sub(home_link, homepage)
        (dest / "_index.md").write_text(homepage, encoding="utf-8")

    for source in docs.rglob("*"):
        if (source.is_file() and source.suffix.lower() != ".md"
                and source.relative_to(docs).parts[0] != "_site"):
            where = dest / "docs" / source.relative_to(docs)
            where.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, where)
    print(f"Generated {len(files)} pages from docs/ into {dest}")
    return len(files)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path)
    opt = ap.parse_args()
    build(opt.root, opt.output)
