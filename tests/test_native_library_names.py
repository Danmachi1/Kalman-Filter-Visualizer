"""Source-consistency checks; these do not execute or validate native filters."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeLibraryNamesTest(unittest.TestCase):
    def test_loader_and_lookup_use_the_same_library(self):
        for name in ("EKFWrapper", "SREKFWrapper", "UKFWrapper", "SRUKFWrapper"):
            with self.subTest(wrapper=name):
                source = (ROOT / f"src/main/java/com/filters/{name}.java").read_text()
                library = re.search(r'LIB_NAME\s*=\s*"([^"]+)"', source).group(1)
                argument = re.search(r'System\.loadLibrary\(([^)]+)\)', source).group(1).strip()
                loaded = library if argument == "LIB_NAME" else argument.strip('"')
                self.assertEqual(library, loaded)
                self.assertIn("SymbolLookup.libraryLookup(LIB_NAME", source)
                self.assertTrue((ROOT / f"{library}.dll").is_file())


if __name__ == "__main__":
    unittest.main()
