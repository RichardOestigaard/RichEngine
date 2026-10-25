"""Staged `richengine serve` options for the qwen4exp stack (Qwen3.8-Flash-Next
today, the Qwen 4 family at release), ported from upstream's
server/serve_options.py.

Not imported anywhere: server/serve_options.py stays untouched — flags that
parse but the engine cannot yet honor are worse than no flags. To integrate,
splice parse_ple_memory and parse_sparse_kv_memory beside the other parsers
and the three ServeOption rows into SERVE_OPTIONS' "memory" group, then wire
the values through the launcher like --max-memory. Upstream folds YaRN into
--max-context (context past the 256K native window turns it on); this tree's
--max-context help text gains the same sentence when the engine reads
yarnContextTokens.
"""


def parse_ple_memory(value):
    return _memory_size(value, "2G", ("off",))


def parse_sparse_kv_memory(value):
    return _memory_size(value, "1200M", ())


QWEN4_SERVE_OPTIONS = [
    ServeOption(
        "--ple-memory",
        "memory",
        dict(
            type=parse_ple_memory,
            default=None,
            metavar="SIZE",
            help="RAM row cache of the PLE n-gram table of a Qwen3.8-Flash-Next "
            "GGUF, e.g. 2G, which streams the rest from the SSD; off loads the "
            "whole table, tens of GB (default: auto, which streams it where "
            "memory cannot hold it)",
        ),
    ),
    ServeOption(
        "--sparse-kv-memory",
        "memory",
        dict(
            type=parse_sparse_kv_memory,
            default=None,
            metavar="SIZE",
            help="RAM cache of the KV blocks a block-sparse model such as "
            "Qwen3.8-Flash-Next reads of requests whose KV is on the SSD, e.g. "
            "1200M; never below its floor (default: auto, its floor)",
        ),
    ),
]
