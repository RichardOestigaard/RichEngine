# Model storage

RichEngine keeps installed models in one models root. Downloads pass through
the Hugging Face cache first, then the installer links or copies the files the
installation owns into that root.

## Where the models root comes from

The root resolves in this order:

1. `--models-dir` on `richengine serve` or `--models` on `richengine models`.
2. `RICHENGINE_MODELS_DIR`.
3. `models_dir` in `~/Library/Application Support/RichEngine/config.json`.
4. The default: a source checkout's existing `install/models`, otherwise
   `~/Library/Application Support/RichEngine/models`.

Inspect the resolution with:

```bash
richengine models root
richengine doctor
```

## Layout

A normal install is a selection link named `OWNER/REPO[:VARIANT]`. Installs
made with source options — a revision, draft override, language-only mode,
thin mode or a local model directory — use `.selections/<hash>` instead. Both
point at immutable entries under these dot-directories:

- `.resolved/`: assemblies of source links and `model.json` records.
- `.blobs/`: content-addressed weight and metadata files the store owns.
- `.packed/`: locally packed package payloads.
- `.metadata/`: metadata derived from GGUF files.

`richengine disk` shows `.selections` installs as `OWNER/REPO (options)` and
prints the `.selections/<hash>` name beside them. Use that name when more
than one option install exists for the same repository.

## Moving the root

Move the store and record the new location with:

```bash
richengine models migrate /Volumes/Models/RichEngine
```

Migration copies with hardlinks where the filesystem permits. On the same
volume, old and new files share inodes: removing the old root does not free
bytes still linked from the new root. Cross-volume migration copies files and
prints progress. The copied installations are verified before `config.json`
changes. The old root is left in place; remove it manually after a successful
serve from the new root.

## Verifying and removing installs

```bash
richengine models verify OWNER/REPO[:VARIANT]
richengine models verify --selection .selections/<hash>
richengine models rm OWNER/REPO[:VARIANT]
richengine models rm .selections/<hash>
```

For an install made with options, pass the same `--revision`, `--draft-model`,
`--language-only` or `--thin` options, or use its exact `.selections/<hash>`
name. Removal deletes the selection link, retires its Hub-cache pins and
collects assemblies or blobs no other install or running server uses.

## Reclaiming disk space

`richengine disk` lists RichEngine, Hugging Face and other local model stores.
Sizes are expensive to walk, so the default says `(size skipped — use
--calc)`; `richengine disk --calc` measures them.

```bash
richengine disk gc          # plan, including the largest dead repos/revisions
richengine disk gc --yes    # collect and report bytes by category
richengine disk dedupe      # find duplicate models across local stores
richengine disk wipe STORE [MODEL] --yes
```

The legacy store keys `splash` and `splash-cache` remain valid. The aliases
`richengine` and `richengine-cache` name the same stores. To remove one
option install through disk wipe, target its exact selection:

```bash
richengine disk wipe richengine .selections/<hash> --yes
```

## Thin installs and the packed cache

`--thin` is experimental and only supports MLX affine targets. The native
engine packs serving weights into a shared cache at
`~/Library/Caches/RichEngine/packed` (or `RICHENGINE_PACKED_CACHE_ROOT`; set
it empty to disable the cache). The target payload links stay until the first
successful serve validates that cache; the next start drops them and reports
how many links were dropped.

The packed cache is derived data. If it is deleted or damaged, verification
fails and the next prepare downloads the recorded target files again, rebuilds
the assembly and repacks the cache. Small metadata, tokenizer, draft and
vision files remain in the models root either way.

Locally packed package installs normally release their upstream source
snapshots because their `.packed` payload is self-contained. Set
`RICHENGINE_PACK_KEEP_SOURCES=1` before installing to retain those snapshots
for offline repair.

## Hugging Face cache

Downloads use the Hugging Face cache. Set `HF_HOME` to move the whole
Hugging Face home directory, including its default `hub` cache. Model links
and agent sessions stay in RichEngine's data directory; existing downloads are
not moved.
