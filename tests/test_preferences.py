import unittest

from recommender import GameSearchEngine


class PreferenceExtractionTests(unittest.TestCase):
    def test_detects_explicit_preferences(self) -> None:
        cases = [
            ("I want a free game", "free"),
            ("no cost horror game", "free"),
            ("cheap indie horror", "cheap"),
            ("low price RPG", "cheap"),
            ("PC strategy game", "windows"),
            ("mac exclusive game", "mac"),
            ("linux native game", "linux"),
            ("steam deck compatible", "linux"),
            ("new action RPG", "recent"),
            ("classic retro RPG", "classic"),
        ]

        for query, preference in cases:
            with self.subTest(query=query, preference=preference):
                result = GameSearchEngine._extract_preferences(query)
                self.assertTrue(result[preference])

    def test_avoids_substring_false_positives(self) -> None:
        cases = [
            ("free roam open world", "free"),
            ("freedom fighter game", "free"),
            ("space combat game", "windows"),
            ("compact city builder", "windows"),
            ("macabre puzzle game", "mac"),
        ]

        for query, preference in cases:
            with self.subTest(query=query, preference=preference):
                result = GameSearchEngine._extract_preferences(query)
                self.assertFalse(result[preference])

    def test_respects_negated_free_constraint(self) -> None:
        for query in ("not free", "isn't free", "without free-to-play games"):
            with self.subTest(query=query):
                result = GameSearchEngine._extract_preferences(query)
                self.assertFalse(result["free"])


if __name__ == "__main__":
    unittest.main()
