"""Parity between the Python and C++ model-family registries.

install/families.py's FAMILIES and the runtime's ModelRegistry
registrations (the registerModelFamily rows in runtime/model/families/*.mm,
what kModelFamilies was) name the same served families and the model_type
each states; drift between them breaks loads. Pure text parsing: no Hub
access, no weights.
"""

import re
import unittest
from pathlib import Path

from install import families, legacy, pack

FAMILIES_DIR = Path(__file__).resolve().parents[2] / "runtime/model/families"

# One registerModelFamily row: {.name = "Name", .modelType = "model_type",
# .legacyType = ""-or-"legacy", ...}. Anchored at line start so a commented
# row does not parse.
ROW = re.compile(
    r'^\s*registerModelFamily\(\s*\{.*?\.name\s*=\s*"([^"]+)".*?'
    r'\.modelType\s*=\s*"([^"]+)".*?\.legacyType\s*=\s*"([^"]*)"',
    re.MULTILINE | re.DOTALL,
)


def cpp_families():
    """Each registerModelFamily row's (model_type, legacy_type) by family name."""
    return {
        name: (model_type, legacy or None)
        for source in FAMILIES_DIR.glob("*.mm")
        for name, model_type, legacy in ROW.findall(source.read_text())
    }


class FamilyParityTest(unittest.TestCase):
    def test_cpp_and_python_name_the_same_families(self):
        # The registry may additionally carry Legacy-alias rows for
        # retired family names — installed packages still resolve; the
        # installer never offers them (families.LEGACY_NAMES).
        python = {family.name for family in families.FAMILIES}
        cpp = set(cpp_families())
        self.assertEqual(
            python | set(families.LEGACY_NAMES),
            cpp,
            f"Python-only: {sorted(python - cpp)}; C++-only: {sorted(cpp - python)}",
        )
        table = cpp_families()
        for alias, target in families.LEGACY_NAMES.items():
            self.assertEqual(
                table[alias][0],
                table[target][0],
                f"legacy alias {alias} states a different model_type than {target}",
            )

    def test_each_signature_states_the_cpp_model_type(self):
        cpp = cpp_families()
        for family in families.FAMILIES:
            with self.subTest(family=family.name):
                signature = dict(family.signature)
                self.assertIn("model_type", signature)
                model_type, legacy_type = cpp[family.name]
                self.assertIn(
                    signature["model_type"],
                    {model_type, legacy_type},
                    f"the registry states {model_type}"
                    + (f" or legacy {legacy_type}" if legacy_type else ""),
                )

    def test_packed_format_defaults_agree_with_pack(self):
        # A family's defaults.packed_format is the target format pack.py
        # writes for its model_type — and only a packable model_type has one.
        for family in families.FAMILIES:
            with self.subTest(family=family.name):
                model_type = dict(family.signature)["model_type"]
                expected = pack.PACKABLE.get(model_type, (None, None))[1]
                self.assertEqual(family.defaults.packed_format, expected)

    def test_package_formats_name_known_families(self):
        for format_name, package in legacy.PACKAGE_FORMATS.items():
            with self.subTest(format=format_name):
                self.assertIsNotNone(
                    families.named(package.family),
                    f"{format_name} names unknown family {package.family}",
                )


if __name__ == "__main__":
    unittest.main()
