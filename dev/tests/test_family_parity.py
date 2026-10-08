"""Parity between the Python and C++ model-family registries.

install/families.py's FAMILIES and runtime's ModelDescriptor.mm
kModelFamilies table name the same served families and the model_type each
states; drift between them breaks loads. Pure text parsing: no Hub access,
no weights.
"""

import re
import unittest
from pathlib import Path

from install import families, legacy, pack

DESCRIPTOR = Path(__file__).resolve().parents[2] / "runtime/model/ModelDescriptor.mm"

# One kModelFamilies row: {"Name", "model_type", nullptr-or-"legacy", make}.
ROW = re.compile(r'\{"([^"]+)",\s*"([^"]+)",\s*(nullptr|"[^"]*")')


def cpp_families():
    """Each kModelFamilies row's (model_type, legacy_type) by family name."""
    source = DESCRIPTOR.read_text()
    start = source.index("kModelFamilies[]")
    table = source[start : source.index("};", start)]
    return {
        name: (model_type, None if legacy == "nullptr" else legacy.strip('"'))
        for name, model_type, legacy in ROW.findall(table)
    }


class FamilyParityTest(unittest.TestCase):
    def test_cpp_and_python_name_the_same_families(self):
        python = {family.name for family in families.FAMILIES}
        cpp = set(cpp_families())
        self.assertEqual(
            python,
            cpp,
            f"Python-only: {sorted(python - cpp)}; C++-only: {sorted(cpp - python)}",
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
                    f"kModelFamilies states {model_type}"
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
