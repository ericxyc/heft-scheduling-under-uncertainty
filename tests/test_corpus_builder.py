"""Tests for expanded-corpus partitioning without network access."""

from __future__ import annotations

import unittest

from heft_reproduction.corpus_cli import _split


class CorpusBuilderTests(unittest.TestCase):
    def test_every_family_has_instance_disjoint_three_way_split(self) -> None:
        splits = [_split(index, 20) for index in range(20)]
        self.assertEqual(splits.count("train"), 14)
        self.assertEqual(splits.count("validation"), 3)
        self.assertEqual(splits.count("test"), 3)

    def test_minimum_family_size_still_has_all_splits(self) -> None:
        self.assertEqual(
            [_split(index, 3) for index in range(3)],
            ["train", "validation", "test"],
        )


if __name__ == "__main__":
    unittest.main()
