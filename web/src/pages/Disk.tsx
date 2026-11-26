import { createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { api, ApiError, apiPost, errorText } from "../api";
import Confirm from "../Confirm";
import { useCopied } from "../clipboard";
import { fmtBytes, fmtInt } from "../format";
import { poll } from "../poll";
import { pushToast } from "../toast";
import "./disk.css";

type StoreModel = { name: string; bytes: number | null };
type Store = {
  name: string;
  title: string;
  paths: string[];
  bytes: number | null;
  exists: boolean;
  locked: boolean;
  models: StoreModel[];
};
type PackedCacheInfo = { path?: string; exists?: boolean; bytes?: number };
type DiskResponse = {
  stores?: Store[];
  free_bytes?: number | null;
  /* Where installs land and where packed weight images cache — present
     only on servers new enough to expose them. */
  models_root?: string;
  packed_cache?: PackedCacheInfo;
};
type WipeResult = { wiped: string; paths: string[]; bytes: number };

/* /v1/disk/gc plan shapes. The endpoint is newer than this page's other
   consumers and its exact shape may drift, so every field reads
   tolerantly: byte maps arrive as {path: bytes} or as lists. */
type GcPlan = {
  cache_dir?: string;
  repos?: number;
  live_revisions?: number;
  dead_repos?: unknown;
  dead_revisions?: unknown;
  shared_blobs?: unknown;
  incomplete?: unknown;
  owned_blobs?: unknown;
  packed_cache?: unknown;
};
type GcApplyResult = { freed?: number; collected?: number };

type DedupeEntry = {
  store: string;
  title: string;
  name: string;
  bytes: number | null;
  weak?: boolean;
};
type DedupeGroup = {
  model: string;
  keep: DedupeEntry[];
  remove: DedupeEntry[];
  reclaimable: number;
  variant: boolean;
};
type DedupePreview = { groups?: DedupeGroup[]; reclaimable?: number };
type DedupeApply = { removed?: string[]; freed?: number };

type StatusDisk = {
  capacity_bytes: number;
  used_bytes: number;
  file_bytes: number;
  read_bytes: number;
  written_bytes: number;
  kv_blocks: number;
  kv_bytes: number;
  kv_demotions: number;
  kv_demotion_failures: number;
  kv_demotions_refused: number;
  kv_restores: number;
  kv_restore_failures: number;
  kv_pending_pages: number;
  persistent: boolean;
  kv_copies: number;
  kv_copy_failures: number;
  taken_back: { states: number; kv_blocks: number; bytes: number; left_behind: number };
  write_behind: { waiting: number; durable: number; unneeded: number; refused: number };
};
type StatusResponse = { disk?: StatusDisk | null };

/* One name+size row of a gc plan section. Byte maps arrive as
   {path: bytes}; lists of paths or names carry no size. */
type GcItem = { name: string; bytes: number | null };

function gcItems(value: unknown, basename = true): GcItem[] {
  /* Byte-map keys and list entries are paths — the basename is the readable
     part. Repo ids (owner/repo) are names already: keep them whole. */
  const base = (path: string) =>
    basename ? path.split("/").pop() || path : path;
  if (Array.isArray(value))
    return value
      .map((entry) => {
        if (typeof entry === "string") return { name: base(entry), bytes: null };
        if (entry && typeof entry === "object") {
          const record = entry as Record<string, unknown>;
          const name = record.name ?? record.path ?? record.repo ?? "";
          if (typeof name === "string" && name)
            return {
              name: base(name),
              bytes: typeof record.bytes === "number" ? record.bytes : null,
            };
        }
        return null;
      })
      .filter((item): item is GcItem => item !== null);
  if (value && typeof value === "object")
    return Object.entries(value as Record<string, unknown>).map(
      ([path, bytes]) => ({
        name: base(path),
        bytes: typeof bytes === "number" ? bytes : null,
      })
    );
  return [];
}

function gcCount(value: unknown): number {
  if (Array.isArray(value)) return value.length;
  if (value && typeof value === "object") return Object.keys(value).length;
  return 0;
}

function gcBytes(value: unknown): number {
  return gcItems(value).reduce((sum, item) => sum + (item.bytes ?? 0), 0);
}

/* The destructive action waiting on AlertDialog confirmation. */
type ConfirmTarget =
  | { kind: "store"; store: Store }
  | { kind: "model"; store: Store; model: StoreModel }
  | { kind: "dedupe"; preview: DedupePreview }
  | { kind: "gc"; plan: GcPlan };

/* Module-level so a 404 from an outdated server retires the GC section for
   the session — a component-local signal would re-scan (and re-404) on every
   page mount. */
const [gcGone, setGcGone] = createSignal(false);

export default function Disk() {
  const [stores, setStores] = createSignal<Store[] | null>(null);
  const [freeBytes, setFreeBytes] = createSignal<number | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [loadError, setLoadError] = createSignal<string | null>(null);
  /* Which long action is running — scan, wipe or apply. Only that control
     shows a busy label; the rest just disable. */
  const [busy, setBusy] = createSignal<null | "scan" | "wipe" | "apply" | "gc" | "gc-apply">(null);
  const { copied, copy } = useCopied();
  const [diskStatus, setDiskStatus] = createSignal<StatusDisk | null>(null);
  const [variants, setVariants] = createSignal(false);
  const [dedupe, setDedupe] = createSignal<DedupePreview | null>(null);
  const [modelsRoot, setModelsRoot] = createSignal<string | null>(null);
  const [packedCache, setPackedCache] = createSignal<PackedCacheInfo | null>(null);
  /* null = not scanned yet; the section hides altogether when the endpoint
     404s — servers predating /v1/disk/gc have nothing to show. */
  const [gc, setGc] = createSignal<GcPlan | null>(null);
  const [query, setQuery] = createSignal("");
  const [sort, setSort] = createSignal<"size" | "name">("size");
  // Pending destructive action awaiting AlertDialog confirmation.
  const [confirm, setConfirm] = createSignal<ConfirmTarget | null>(null);

  async function loadDisk() {
    try {
      // The endpoint walks every store's files; a large cache takes a while.
      const data = await api<DiskResponse>("/v1/disk", {}, 120_000);
      setStores(data.stores ?? []);
      setFreeBytes(data.free_bytes ?? null);
      setModelsRoot(data.models_root ?? null);
      setPackedCache(data.packed_cache ?? null);
      setLoadError(null);
    } catch (error) {
      setLoadError(errorText(error));
    } finally {
      setLoading(false);
    }
  }

  async function pollStatus() {
    try {
      const data = await api<StatusResponse>("/status");
      setDiskStatus(data.disk ?? null);
    } catch {
      /* Telemetry is best-effort: keep the last reading. */
    }
  }

  onMount(() => {
    loadDisk();
  });
  onCleanup(poll(pollStatus, 5000));

  const existingCount = () => stores()?.filter((store) => store.exists).length ?? 0;
  const totalCount = () => stores()?.length ?? 0;
  const noStores = () => {
    const list = stores();
    return list !== null && (list.length === 0 || list.every((store) => !store.exists));
  };
  const modelCount = () =>
    stores()?.reduce((sum, store) => sum + (store.models?.length ?? 0), 0) ?? 0;
  /* Models surviving the search filter, across all stores. */
  const matchCount = () =>
    (stores() ?? []).reduce((sum, store) => sum + viewModels(store).length, 0);

  /* Models shown inside a store after the search filter and sort apply. */
  const viewModels = (store: Store) => {
    const q = query().trim().toLowerCase();
    const list = (store.models ?? []).filter(
      (model) => !q || model.name.toLowerCase().includes(q)
    );
    return sort() === "name"
      ? [...list].sort((a, b) => a.name.localeCompare(b.name))
      : [...list].sort((a, b) => (b.bytes ?? 0) - (a.bytes ?? 0));
  };

  /* Bytes a gc plan accounts for exactly: the byte-mapped sections.
     Dead revisions and partial downloads inside the Hub cache add more,
     but their sizes come only from the delete strategy at apply time. */
  const gcPlanBytes = (plan: GcPlan) =>
    gcBytes(plan.shared_blobs) +
    gcBytes(plan.owned_blobs) +
    gcBytes(plan.packed_cache);

  const gcEmpty = (plan: GcPlan) =>
    gcCount(plan.dead_repos) === 0 &&
    gcCount(plan.dead_revisions) === 0 &&
    gcCount(plan.shared_blobs) === 0 &&
    gcCount(plan.incomplete) === 0 &&
    gcCount(plan.owned_blobs) === 0 &&
    gcCount(plan.packed_cache) === 0;

  /* Bytes the pending confirmation would free: the card projects the free
     space afterwards while the dialog is open. */
  const pendingBytes = () => {
    const target = confirm();
    if (!target) return 0;
    if (target.kind === "store") return target.store.bytes ?? 0;
    if (target.kind === "model") return target.model.bytes ?? 0;
    if (target.kind === "gc") return gcPlanBytes(target.plan);
    return target.preview.reclaimable ?? 0;
  };

  async function runWipe(body: Record<string, unknown>, done: (bytes: number | null) => string) {
    if (busy()) return;
    setBusy("wipe");
    try {
      const result = await apiPost<WipeResult>("/v1/disk/wipe", { ...body, confirm: true });
      pushToast("ok", done(result.bytes ?? null));
      // The dedupe preview references wiped paths — drop it.
      setDedupe(null);
      await loadDisk();
    } catch (error) {
      pushToast("error", errorText(error));
    } finally {
      setBusy(null);
    }
  }

  /* Runs whichever action the confirm dialog is currently describing. */
  function runConfirmed() {
    const target = confirm();
    if (!target) return;
    setConfirm(null);
    if (target.kind === "store") {
      void runWipe(
        { store: target.store.name },
        (bytes) => `Deleted ${target.store.title}: freed ${fmtBytes(bytes)}.`
      );
    } else if (target.kind === "model") {
      void runWipe(
        { store: target.store.name, model: target.model.name },
        (bytes) => `Deleted ${target.model.name} from ${target.store.title}: freed ${fmtBytes(bytes)}.`
      );
    } else if (target.kind === "gc") {
      void applyGc();
    } else {
      void applyDedupe(target.preview);
    }
  }

  const confirmTitle = () => {
    const target = confirm();
    if (!target) return "";
    if (target.kind === "store") return `Delete ${target.store.title}?`;
    if (target.kind === "model") return `Delete ${target.model.name}?`;
    if (target.kind === "gc") return "Collect unreferenced cache?";
    return "Remove duplicates?";
  };

  const confirmLabel = () => {
    const target = confirm();
    if (!target) return "Confirm";
    if (target.kind === "store") return "Delete store";
    if (target.kind === "model") return "Delete";
    if (target.kind === "gc") return `Reclaim ${fmtBytes(gcPlanBytes(target.plan))}`;
    return `Reclaim ${fmtBytes(target.preview.reclaimable ?? null)}`;
  };

  const confirmDesc = () => {
    const target = confirm();
    if (!target) return "";
    if (target.kind === "store")
      return `This permanently deletes every model file in ${target.store.title} — ${fmtBytes(target.store.bytes)} reclaimed. This cannot be undone.`;
    if (target.kind === "model")
      return `This permanently deletes ${target.model.name} from ${target.store.title} — ${fmtBytes(target.model.bytes)} reclaimed. This cannot be undone.`;
    if (target.kind === "gc") {
      const parts: string[] = [];
      const repos = gcCount(target.plan.dead_repos);
      const revisions = gcCount(target.plan.dead_revisions);
      const blobs = gcCount(target.plan.shared_blobs);
      const partial = gcCount(target.plan.incomplete);
      const owned = gcCount(target.plan.owned_blobs);
      const packed = gcCount(target.plan.packed_cache);
      if (repos) parts.push(`${fmtInt(repos)} unreferenced ${repos === 1 ? "repo" : "repos"}`);
      if (revisions) parts.push(`${fmtInt(revisions)} dead ${revisions === 1 ? "revision" : "revisions"}`);
      if (blobs) parts.push(`${fmtInt(blobs)} shared ${blobs === 1 ? "blob" : "blobs"}`);
      if (partial) parts.push(`${fmtInt(partial)} partial ${partial === 1 ? "download" : "downloads"}`);
      if (owned) parts.push(`${fmtInt(owned)} orphaned model ${owned === 1 ? "blob" : "blobs"}`);
      if (packed) parts.push(`${fmtInt(packed)} packed cache ${packed === 1 ? "entry" : "entries"}`);
      return `This permanently deletes cached weight data nothing references — ${parts.join(", ") || "nothing"}. Everything an installed model links to survives; at least ${fmtBytes(gcPlanBytes(target.plan))} reclaimed. This cannot be undone.`;
    }
    /* Weak matches are skipped by apply — don't count them as deletions. */
    const dupes =
      target.preview.groups?.reduce(
        (n, g) => n + (g.remove?.filter((e) => !e.weak).length ?? 0),
        0
      ) ?? 0;
    return `This permanently deletes ${fmtInt(dupes)} duplicate ${
      dupes === 1 ? "copy" : "copies"
    } marked "remove" across the stores — one copy of each model is kept. ${fmtBytes(
      target.preview.reclaimable ?? null
    )} reclaimed. This cannot be undone.`;
  };

  /* GC mirrors the dedupe flow: scan is a read-only plan, apply deletes
     behind the confirm dialog. The endpoint postdates some servers — a
     404 retires the section instead of erroring on every scan. */
  async function scanGc() {
    if (busy()) return;
    setBusy("gc");
    setGc(null);
    try {
      const data = await apiPost<GcPlan>("/v1/disk/gc", { apply: false });
      setGc(data);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        setGcGone(true);
      } else {
        pushToast("error", errorText(error));
      }
    } finally {
      setBusy(null);
    }
  }

  async function applyGc() {
    if (busy()) return;
    setBusy("gc-apply");
    try {
      const result = await apiPost<GcApplyResult>("/v1/disk/gc", {
        apply: true,
        confirm: true,
      });
      const freed = result.freed ?? result.collected ?? null;
      pushToast("ok", `Collected unreferenced cache: freed ${fmtBytes(freed)}.`);
      // The plan is stale after collection — clear it and refresh.
      setGc(null);
      await loadDisk();
    } catch (error) {
      pushToast("error", errorText(error));
    } finally {
      setBusy(null);
    }
  }

  async function scanDedupe() {
    if (busy()) return;
    setBusy("scan");
    setDedupe(null);
    try {
      const data = await apiPost<DedupePreview>("/v1/disk/dedupe", {
        variants: variants(),
        apply: false,
      });
      setDedupe(data);
    } catch (error) {
      pushToast("error", errorText(error));
    } finally {
      setBusy(null);
    }
  }

  async function applyDedupe(preview: DedupePreview) {
    if (busy()) return;
    setBusy("apply");
    try {
      const result = await apiPost<DedupeApply>("/v1/disk/dedupe", {
        variants: variants(),
        apply: true,
        confirm: true,
      });
      const removed = result.removed?.length ?? 0;
      pushToast(
        "ok",
        `Removed ${removed} duplicate${removed === 1 ? "" : "s"}: freed ${fmtBytes(result.freed ?? null)}.`
      );
      // Preview is stale after removals — clear it and refresh the stores.
      setDedupe(null);
      await loadDisk();
    } catch (error) {
      pushToast("error", errorText(error));
    } finally {
      setBusy(null);
    }
  }

  const meterPercent = () => {
    const disk = diskStatus();
    if (!disk || !disk.capacity_bytes) return 0;
    return Math.min(100, Math.max(0, (disk.used_bytes / disk.capacity_bytes) * 100));
  };

  return (
    <main class="page disk-page">
      <h1>Disk</h1>
      <p class="lede">
        Model storage across local tools — wipe to reclaim space, dedupe to remove cross-store duplicates.
      </p>

      <div class="cards">
        <div class="card">
          <div class="card-label">Free space</div>
          <div class="card-value">{fmtBytes(freeBytes())}</div>
          <div class="card-sub">
            {pendingBytes() > 0 && freeBytes() !== null
              ? `≈ ${fmtBytes((freeBytes() ?? 0) + pendingBytes())} once confirmed`
              : "available on this Mac"}
          </div>
        </div>
        <div class="card">
          <div class="card-label">Stores</div>
          <div class="card-value">
            {stores() === null ? "—" : `${existingCount()} / ${totalCount()}`}
          </div>
          <div class="card-sub">found / configured</div>
        </div>
        <div class="card">
          <div class="card-label">Models</div>
          <div class="card-value">{stores() === null ? "—" : fmtInt(modelCount())}</div>
          <div class="card-sub">cached across stores</div>
        </div>
      </div>

      <Show when={loading() && stores() === null}>
        <div class="panel">
          <div class="panel-row">
            <span class="grow muted">Scanning disk…</span>
          </div>
        </div>
      </Show>

      {/* Storage locations — feature-detected: only servers exposing
          models_root/packed_cache in /v1/disk render this. */}
      <Show when={modelsRoot() || packedCache()?.path}>
        <section class="section">
          <h2>Locations</h2>
          <div class="panel">
            <Show when={modelsRoot()}>
              {(root) => (
                <div class="panel-row">
                  <span class="grow">Models root</span>
                  <button
                    type="button"
                    class="copy-btn mono muted location-path"
                    title={`Copy ${root()}`}
                    onClick={() => void copy("models-root", root())}
                  >
                    {copied() === "models-root" ? "copied ✓" : root()}
                  </button>
                </div>
              )}
            </Show>
            <Show when={packedCache()?.path}>
              <div class="panel-row">
                <span class="grow">
                  Packed weight cache
                  <Show when={packedCache() && !packedCache()?.exists}>
                    {" "}
                    <span class="badge" title="Nothing cached yet">empty</span>
                  </Show>
                </span>
                <span class="num">{fmtBytes(packedCache()?.bytes ?? null)}</span>
                <button
                  type="button"
                  class="copy-btn mono muted location-path"
                  title={`Copy ${packedCache()?.path ?? ""}`}
                  onClick={() => void copy("packed-cache", packedCache()?.path ?? "")}
                >
                  {copied() === "packed-cache" ? "copied ✓" : packedCache()?.path}
                </button>
              </div>
            </Show>
          </div>
        </section>
      </Show>

      <Show when={loadError()} keyed>
        {(text: string) => (
          <p class="notice error">
            {text}{" "}
            <button class="btn small" onClick={() => void loadDisk()}>
              Retry
            </button>
          </p>
        )}
      </Show>

      <section class="section">
        <h2>Duplicates</h2>
        <div class="dedupe-controls">
          <button
            class="btn"
            title="Find the same model stored in more than one place"
            disabled={!!busy() || loading()}
            onClick={scanDedupe}
          >
            {busy() === "scan" ? "Scanning…" : "Scan for duplicates"}
          </button>
          <label class="dedupe-variants muted">
            <input
              type="checkbox"
              title="Also group format variants (e.g. MLX vs GGUF) of the same model"
              checked={variants()}
              disabled={!!busy()}
              onChange={(event) => {
                setVariants(event.currentTarget.checked);
                setDedupe(null);
              }}
            />
            include format variants
          </label>
        </div>

        <Show when={dedupe()} keyed>
          {(preview: DedupePreview) => (
            <>
              <Show when={(preview.groups?.length ?? 0) === 0}>
                <p class="muted">No duplicate models across stores.</p>
              </Show>
              <For each={preview.groups ?? []}>
                {(group) => (
                  <div class="panel dedupe-group">
                    <div class="panel-row dedupe-head">
                      <span class="grow mono">{group.model}</span>
                      <Show when={group.variant}>
                        <span class="badge warn">formats</span>
                      </Show>
                      <span class="badge">{fmtBytes(group.reclaimable)} reclaimable</span>
                    </div>
                    <For each={group.keep ?? []}>
                      {(entry) => (
                        <div class="panel-row">
                          <span class="badge ok">keep</span>
                          <span class="grow">
                            {entry.title} — {entry.name}
                            <Show when={entry.weak}>
                              {" "}
                              <span class="muted">(unverified)</span>
                            </Show>
                          </span>
                          <span class="num">{fmtBytes(entry.bytes)}</span>
                        </div>
                      )}
                    </For>
                    <For each={group.remove ?? []}>
                      {(entry) => (
                        <div class="panel-row" classList={{ weak: !!entry.weak }}>
                          <Show
                            when={entry.weak}
                            fallback={<span class="badge">remove</span>}
                          >
                            <span
                              class="badge warn"
                              title="Only a weak match — this copy is kept"
                            >
                              skip
                            </span>
                          </Show>
                          <span class="grow">
                            {entry.title} — {entry.name}
                            <Show when={entry.weak}>
                              {" "}
                              <span class="muted">(unverified — kept)</span>
                            </Show>
                          </span>
                          <span class="num">{fmtBytes(entry.bytes)}</span>
                        </div>
                      )}
                    </For>
                  </div>
                )}
              </For>
              <Show when={(preview.groups?.length ?? 0) > 0}>
                <div class="panel dedupe-footer">
                  <div class="panel-row">
                    <span class="grow muted">
                      {fmtInt(preview.groups?.length)} duplicate{preview.groups?.length === 1 ? "" : "s"} ·{" "}
                      {fmtBytes(preview.reclaimable)} reclaimable — one copy kept
                      per model
                    </span>
                    <button
                      class="btn danger"
                      title="Delete the copies marked remove — keeps one per model"
                      disabled={!!busy()}
                      onClick={() => setConfirm({ kind: "dedupe", preview })}
                    >
                      {`Reclaim ${fmtBytes(preview.reclaimable)}`}
                    </button>
                  </div>
                </div>
              </Show>
            </>
          )}
        </Show>
      </section>

      {/* Garbage collection — scan plans, confirm applies. The whole
          section retires when the server predates /v1/disk/gc. */}
      <Show when={!gcGone()}>
        <section class="section">
          <h2>Garbage collection</h2>
          <div class="dedupe-controls">
            <button
              class="btn"
              title="Walk the Hub cache, model blobs and packed weight cache for entries no installed model references"
              disabled={!!busy() || loading()}
              onClick={scanGc}
            >
              {busy() === "gc" ? "Scanning…" : "Scan for collectable cache"}
            </button>
            <span class="muted">
              drops cached weight data nothing references — everything an
              installed model links to survives
            </span>
          </div>
          <Show when={gc()} keyed>
            {(plan: GcPlan) => (
              <>
                <Show when={gcEmpty(plan)}>
                  <p class="muted">
                    Nothing to collect — every cached file is referenced by an
                    installed model.
                  </p>
                </Show>
                {/* Counts first; the byte-mapped sections list their entries
                    below. */}
                <Show when={!gcEmpty(plan)}>
                  <div class="panel gc-group">
                    <div class="panel-row dedupe-head">
                      <span class="grow">Plan</span>
                      <span class="badge">
                        {fmtBytes(gcPlanBytes(plan))}+ reclaimable
                      </span>
                    </div>
                    <Show when={gcCount(plan.dead_repos) > 0}>
                      <div class="panel-row">
                        <span class="grow">Unreferenced repos</span>
                        <span class="num">{fmtInt(gcCount(plan.dead_repos))}</span>
                      </div>
                    </Show>
                    <Show when={gcCount(plan.dead_revisions) > 0}>
                      <div class="panel-row">
                        <span class="grow">Dead revisions</span>
                        <span class="num">{fmtInt(gcCount(plan.dead_revisions))}</span>
                      </div>
                    </Show>
                    <Show when={gcCount(plan.incomplete) > 0}>
                      <div class="panel-row">
                        <span class="grow">Partial downloads</span>
                        <span class="num">{fmtInt(gcCount(plan.incomplete))}</span>
                      </div>
                    </Show>
                  </div>
                  <For
                    each={[
                      ["Unreferenced repos", gcItems(plan.dead_repos, false)],
                      ["Shared blobs", gcItems(plan.shared_blobs)],
                      ["Orphaned model blobs", gcItems(plan.owned_blobs)],
                      ["Packed cache entries", gcItems(plan.packed_cache)],
                    ] as [string, GcItem[]][]}
                  >
                    {([title, items]) => (
                      <Show when={items.length > 0}>
                        <div class="panel gc-group">
                          <div class="panel-row dedupe-head">
                            <span class="grow">{title}</span>
                            <span class="badge">
                              {fmtBytes(
                                items.reduce((sum, item) => sum + (item.bytes ?? 0), 0)
                              )}
                            </span>
                          </div>
                          <For each={items.slice(0, 20)}>
                            {(item) => (
                              <div class="panel-row">
                                <span class="grow mono">{item.name}</span>
                                <span class="num">{fmtBytes(item.bytes)}</span>
                              </div>
                            )}
                          </For>
                          <Show when={items.length > 20}>
                            <div class="panel-row">
                              <span class="grow muted">
                                … {fmtInt(items.length - 20)} more
                              </span>
                            </div>
                          </Show>
                        </div>
                      </Show>
                    )}
                  </For>
                  <div class="panel dedupe-footer">
                    <div class="panel-row">
                      <span class="grow muted">
                        installed models keep everything they reference
                      </span>
                      <button
                        class="btn danger"
                        title="Delete every entry the plan lists"
                        disabled={!!busy()}
                        onClick={() => setConfirm({ kind: "gc", plan })}
                      >
                        {busy() === "gc-apply"
                          ? "Collecting…"
                          : `Reclaim ${fmtBytes(gcPlanBytes(plan))}`}
                      </button>
                    </div>
                  </div>
                </Show>
              </>
            )}
          </Show>
        </section>
      </Show>

      <section class="section">
        <div class="stores-head">
          <h2>Stores</h2>
          <Show when={modelCount() > 3}>
            <input
              class="store-search"
              type="search"
              placeholder="Filter models…"
              aria-label="Filter models"
              value={query()}
              onInput={(event) => setQuery(event.currentTarget.value)}
            />
            <select
              class="store-sort"
              aria-label="Sort models"
              value={sort()}
              onChange={(event) => setSort(event.currentTarget.value as "size" | "name")}
            >
              <option value="size">By size</option>
              <option value="name">By name</option>
            </select>
          </Show>
          <Show when={query().trim()}>
            <span class="muted">{fmtInt(matchCount())} matches</span>
          </Show>
          <button
            class="btn small"
            title="Rescan the stores"
            disabled={loading()}
            onClick={() => void loadDisk()}
          >
            {loading() ? "Scanning…" : "Refresh"}
          </button>
        </div>
        <Show when={busy() === "wipe" || busy() === "apply"}>
          <p class="muted">Working…</p>
        </Show>
        <Show when={noStores()}>
          <div class="panel">
            <div class="panel-row">
              <span class="grow muted">No model stores found</span>
            </div>
          </div>
        </Show>
        <For each={stores() ?? []}>
          {(store) => (
            <Show
              when={store.exists}
              fallback={
                <div class="panel store-panel store-missing">
                  <div class="panel-row">
                    <span class="grow muted">{store.title}</span>
                    <span class="badge" title="The configured path does not exist">
                      not found
                    </span>
                  </div>
                  <For each={store.paths ?? []}>
                    {(path) => (
                      <div class="panel-row store-path">
                        <span class="mono muted grow">{path}</span>
                      </div>
                    )}
                  </For>
                </div>
              }
            >
              <div class="panel store-panel">
                <div class="panel-row store-head">
                  <span class="grow store-title">{store.title}</span>
                  <Show when={store.locked}>
                    <span class="badge warn" title="Stop the server before wiping the store it serves from">
                      in use by this server
                    </span>
                  </Show>
                  <span class="badge">{fmtBytes(store.bytes)}</span>
                  <button
                    class="btn small danger"
                    disabled={!!busy() || store.locked}
                    title={
                      store.locked
                        ? "Stop the server before deleting the store it serves from"
                        : `Delete all of ${store.title}`
                    }
                    onClick={() => setConfirm({ kind: "store", store })}
                  >
                    Delete store
                  </button>
                </div>
                <For each={store.paths ?? []}>
                  {(path) => (
                    <div class="panel-row store-path">
                      <span class="mono muted grow">{path}</span>
                    </div>
                  )}
                </For>
                <For each={viewModels(store)}>
                  {(model) => (
                    <div class="panel-row">
                      <button
                        type="button"
                        class="model-name grow"
                        title="Copy model name"
                        onClick={() =>
                          void copy(`${store.name}/${model.name}`, model.name)
                        }
                      >
                        {model.name}
                        <Show when={copied() === `${store.name}/${model.name}`}>
                          {" "}
                          <span class="muted">✓</span>
                        </Show>
                      </button>
                      <span class="num">{fmtBytes(model.bytes)}</span>
                      <button
                        class="btn small danger"
                        disabled={!!busy() || store.locked}
                        title={
                          store.locked
                            ? "Store is in use by this server"
                            : `Delete ${model.name}`
                        }
                        onClick={() => setConfirm({ kind: "model", store, model })}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </For>
                <Show
                  when={
                    (store.models?.length ?? 0) === 0 ||
                    (query().trim() !== "" && viewModels(store).length === 0)
                  }
                >
                  <div class="panel-row">
                    <span class="grow muted">
                      {(store.models?.length ?? 0) === 0
                        ? "No model entries"
                        : `No matches for "${query().trim()}"`}
                    </span>
                  </div>
                </Show>
              </div>
            </Show>
          )}
        </For>
      </section>

      <Show when={diskStatus()}>
        {(disk) => (
          <section class="section">
            <h2>KV disk cache</h2>
            <Show
              when={disk().capacity_bytes > 0}
              fallback={
                <p class="muted">SSD cache off — enable with --max-cache-disk</p>
              }
            >
              <div class="panel">
                <div class="kv-meter">
                  <div
                    class="meter"
                    role="meter"
                    aria-label="SSD cache used"
                    aria-valuemin={0}
                    aria-valuemax={disk().capacity_bytes}
                    aria-valuenow={disk().used_bytes}
                    aria-valuetext={`${fmtBytes(disk().used_bytes)} of ${fmtBytes(disk().capacity_bytes)}`}
                  >
                    <i style={{ width: `${meterPercent()}%` }} />
                  </div>
                </div>
                <div class="panel-row">
                  <span class="grow">SSD cache</span>
                  <span class="num">
                    {fmtBytes(disk().used_bytes)} of {fmtBytes(disk().capacity_bytes)} ·{" "}
                    {Math.round(meterPercent())}%
                    <Show when={disk().persistent}>
                      {" "}
                      <span class="badge ok">persistent</span>
                    </Show>
                  </span>
                </div>
                <div class="panel-row" title="KV blocks currently stored on SSD; pending pages are queued writes">
                  <span class="grow">Cached KV</span>
                  <span class="num">
                    {fmtBytes(disk().kv_bytes)} in {fmtInt(disk().kv_blocks)} blocks
                    <Show when={disk().kv_pending_pages > 0}>
                      {" "}· {fmtInt(disk().kv_pending_pages)} pages pending
                    </Show>
                  </span>
                </div>
                <div class="panel-row" title="KV blocks moved from RAM to SSD under memory pressure; refused = skipped because pressure resolved first">
                  <span class="grow">Demotions (RAM → SSD)</span>
                  <span class="num">
                    {fmtInt(disk().kv_demotions)}
                    <Show when={disk().kv_demotion_failures > 0}>
                      {" "}· {fmtInt(disk().kv_demotion_failures)} failed
                    </Show>
                    <Show when={disk().kv_demotions_refused > 0}>
                      {" "}· {fmtInt(disk().kv_demotions_refused)} refused
                    </Show>
                  </span>
                </div>
                <div class="panel-row" title="KV blocks read back from SSD into RAM">
                  <span class="grow">Restores (SSD → RAM)</span>
                  <span class="num">
                    {fmtInt(disk().kv_restores)}
                    <Show when={disk().kv_restore_failures > 0}>
                      {" "}· {fmtInt(disk().kv_restore_failures)} failed
                    </Show>
                  </span>
                </div>
                <div class="panel-row" title="Writes buffered before reaching disk — waiting = queued, durable = flushed; refused = skipped">
                  <span class="grow">Write-behind</span>
                  <span class="num">
                    {fmtInt(disk().write_behind?.waiting)} waiting · {fmtInt(disk().write_behind?.durable)} durable
                    <Show when={(disk().write_behind?.refused ?? 0) > 0}>
                      {" "}· {fmtInt(disk().write_behind.refused)} refused
                    </Show>
                  </span>
                </div>
                <div class="panel-row" title="Total bytes read from and written to the SSD cache since start">
                  <span class="grow">SSD read / written</span>
                  <span class="num">
                    {fmtBytes(disk().read_bytes)} / {fmtBytes(disk().written_bytes)}
                  </span>
                </div>
              </div>
            </Show>
          </section>
        )}
      </Show>


      <Confirm
        open={confirm() !== null}
        title={confirmTitle()}
        confirmLabel={confirmLabel()}
        busy={busy() !== null}
        onConfirm={runConfirmed}
        onClose={() => setConfirm(null)}
      >
        {confirmDesc()}
      </Confirm>
    </main>
  );
}
