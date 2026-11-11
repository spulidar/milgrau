"""Ensure the Lotus Docs website never becomes a second documentation source."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "build.py"
SPEC = importlib.util.spec_from_file_location("milgrau_docs_generator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SiteDocumentationTests(unittest.TestCase):
    def test_generated_internal_links_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_text("# Index\n\n[Level 0](libids.md)\n", encoding="utf-8")
            raw = "# LIBIDS\n\nBack to [Index](README.md).\n"
            (docs / "libids.md").write_text(raw, encoding="utf-8")
            self.assertEqual(MODULE.build(root), 2)
            item = (root / "docs/_site/.hugo-content/docs/processing/level-0/libids.md").read_text()
            self.assertIn('relref "docs/_index.md"', item)
            self.assertIn("katex: true", item)
            self.assertNotIn("# LIBIDS\n", item)
            self.assertEqual((docs / "libids.md").read_text(), raw)
            index = (root / "docs/_site/.hugo-content/docs/_index.md").read_text()
            self.assertIn('relref "docs/processing/level-0/libids.md"', index)

    def test_broken_links_fail_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs").mkdir()
            (root / "docs/README.md").write_text("# Index\n\n[Missing](missing.md)\n")
            with self.assertRaisesRegex(ValueError, "Missing linked source"):
                MODULE.build(root)

    def test_existing_unmanaged_output_is_never_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs").mkdir()
            (root / "docs/_site/.hugo-content").mkdir(parents=True)
            (root / "docs/_site/.hugo-content/important.txt").write_text("Keep")
            with self.assertRaisesRegex(ValueError, "unmanaged"):
                MODULE.build(root)
            self.assertTrue((root / "docs/_site/.hugo-content/important.txt").exists())

    def test_build_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs").mkdir()
            (root / "docs/README.md").write_text("# Index\n\nHello\n")
            self.assertEqual(MODULE.build(root), 1)
            output = root / "docs/_site/.hugo-content/docs/_index.md"
            before = output.read_bytes()
            self.assertEqual(MODULE.build(root), 1)
            self.assertEqual(before, output.read_bytes())


    def test_site_infrastructure_and_previous_build_are_not_source_pages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / "docs/_site"
            site.mkdir(parents=True)
            (root / "docs/README.md").write_text("# Scientific documentation\n", encoding="utf-8")
            (site / "README.md").write_text("# Site configuration, not a scientific page\n", encoding="utf-8")
            self.assertEqual(MODULE.build(root), 1)
            self.assertEqual(MODULE.build(root), 1)
            self.assertFalse((site / ".hugo-content/docs/reference/README.md").exists())

    def test_cannot_overwrite_site_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / "docs/_site"
            site.mkdir(parents=True)
            (root / "docs/README.md").write_text("# Original\n", encoding="utf-8")
            (site / "hugo.toml").write_text("Original config", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite source"):
                MODULE.build(root, site)
            self.assertEqual((site / "hugo.toml").read_text(), "Original config")


    def test_reading_order_is_global_and_follows_group_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            for filename in (
                "README.md", "getting_started.md", "operating_milgrau.md",
                "libids.md", "level0_schema.md", "lipancora.md", "lebear.md",
            ):
                (docs / filename).write_text("# " + filename + "\n", encoding="utf-8")
            self.assertEqual(MODULE.build(root), 7)
            output = root / "docs/_site/.hugo-content/docs"
            routes = [
                "getting-started/getting_started.md",
                "getting-started/operating_milgrau.md",
                "processing/level-0/libids.md",
                "processing/level-0/level0_schema.md",
                "processing/level-1/lipancora.md",
                "processing/level-2/lebear.md",
            ]
            weights = []
            for relative in routes:
                body = (output / relative).read_text(encoding="utf-8")
                weight = next(
                    int(line.split(": ", 1)[1])
                    for line in body.splitlines() if line.startswith("weight: ")
                )
                weights.append(weight)
            self.assertEqual(weights, sorted(set(weights)))


    def test_homepage_is_derived_from_readme_without_copying_mermaid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "README.md").write_text("# Documentation\n", encoding="utf-8")
            (docs / "getting_started.md").write_text("# Installation\n", encoding="utf-8")
            readme = (
                "# MILGRAU\n\n"
                "A science pipeline for lidar.\n\n"
                "## Scientific processing workflow\n\n"
                "LIBIDS, LIPANCORA and LEBEAR.\n\n"
                "### Processing pathway at a glance\n\n"
                "```mermaid\nflowchart LR\nA --> B\n```\n\n"
                "### Level 0 — LIBIDS\n\nDetailed method.\n\n"
                "## Installation\n\n"
                "See [Getting started](docs/getting_started.md).\n\n"
                "## Scientific quality and interpretation\n\n"
                "Quality diagnostics are necessary.\n\n"
                "## License\n\nMIT.\n"
            )
            (root / "README.md").write_text(readme, encoding="utf-8")
            MODULE.build(root)
            page = (root / "docs/_site/.hugo-content/_index.md").read_text(encoding="utf-8")
            self.assertIn("A science pipeline for lidar", page)
            self.assertIn("LIBIDS, LIPANCORA and LEBEAR", page)
            self.assertIn("Quality diagnostics are necessary", page)
            self.assertIn('relref "docs/getting-started/getting_started.md"', page)
            self.assertNotIn("```mermaid", page)
            self.assertEqual((root / "README.md").read_text(encoding="utf-8"), readme)



if __name__ == "__main__":
    unittest.main()
