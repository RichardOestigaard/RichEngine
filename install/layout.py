"""The models root's on-disk layout: the names the installers share.

A selection link under the models root (paths.MODELS) is OWNER/REPO[:VARIANT],
or .selections/<hash of the selection> for a selection with source options.
What it links lives in a dot-directory: an assembly is .resolved/<SHA-256 of
its record> (assembly.py), metadata derived from a GGUF is .metadata/<key>
and a locally packed package is .packed/<key> (pack.py). .install.lock
serializes every write, and a directory named by a staging prefix beside an
entry is one an interrupted write left, which garbage collection removes.

The record files and the assembly subtree's directories are fixed names the
runtime reads: model.json records an assembly's sources and linked files,
manifest.json a package's artifacts and files.json a metadata entry's files;
target/, draft/, vision/ and tokenizer/ hold the files each loads.
"""

from __future__ import annotations

# The dot-directories the models root's entries live under.
RESOLVED = ".resolved"
METADATA = ".metadata"
PACKED = ".packed"
SELECTIONS = ".selections"
INSTALL_LOCK = ".install.lock"

# The staging prefixes an interrupted installation leaves, each beside what
# it was writing: a selection link, an assembly or metadata entry, a package.
LINK_STAGING = ".prepare-"
ENTRY_STAGING = ".loading-"
PACK_STAGING = ".packing-"

# The record file of each store entry.
ASSEMBLY_RECORD = "model.json"
PACKAGE_MANIFEST = "manifest.json"
METADATA_RECORD = "files.json"

# The assembly subtree's directories (a packed package's artifacts too).
TARGET = "target"
DRAFT = "draft"
VISION = "vision"
TOKENIZER = "tokenizer"
# The GGUF vision projector's assembly path.
GGUF_VISION = VISION + "/mmproj.gguf"
