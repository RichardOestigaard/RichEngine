import { createSignal, For, onCleanup, onMount, Show } from "solid-js";
import { api, apiPost, errorText } from "../api";
import Confirm from "../Confirm";
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
type DiskResponse = { stores?: Store[]; free_bytes?: number | null };
type WipeResult = { wiped: string; paths: string[]; bytes: number };

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

/* The destructive action waiting on AlertDialog confirmation. */
type ConfirmTarget =
  | { kind: "store"; store: Store }
  | { kind: "model"; store: Store; model: StoreModel }
  | { kind: "dedupe"; preview: DedupePreview };

export default function Disk() {
  const [stores, setStores] = createSignal<Store[] | null>(null);
  const [freeBytes, setFreeBytes] = createSignal<number | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [loadError, setLoadError] = createSignal<string | null>(null);
  const [busy, setBusy] = createSignal(false);
  const [diskStatus, setDiskStatus] = createSignal<StatusDisk | null>(null);
  const [variants, setVariants] = createSignal(false);
  const [dedupe, setDedupe] = createSignal<DedupePreview | null>(null);
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

  /* Bytes the pending confirmation would free: the card projects the free
     space afterwards while the dialog is open. */
  const pendingBytes = () => {
    const target = confirm();
    if (!target) return 0;
    if (target.kind === "store") return target.store.bytes ?? 0;
    if (target.kind === "model") return target.model.bytes ?? 0;
    return target.preview.reclaimable ?? 0;
  };

  async function runWipe(body: Record<string, unknown>, done: (bytes: number | null) => string) {
    if (busy()) return;
    setBusy(true);
    try {
      const result = await apiPost<WipeResult>("/v1/disk/wipe", { ...body, confirm: true });
      pushToast("ok", done(result.bytes ?? null));
      // The dedupe preview references wiped paths — drop it.
      setDedupe(null);
      await loadDisk();
    } catch (error) {
      pushToast("error", errorText(error));
    } finally {
      setBusy(false);
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
        (bytes) => `Wiped ${target.store.title}: freed ${fmtBytes(bytes)}.`
      );
    } else if (target.kind === "model") {
      void runWipe(
        { store: target.store.name, model: target.model.name },
        (bytes) => `Wiped ${target.model.name} from ${target.store.title}: freed ${fmtBytes(bytes)}.`
      );
    } else {
      void applyDedupe(target.preview);
    }
  }

  const confirmTitle = () => {
    const target = confirm();
    if (!target) return "";
    if (target.kind === "store") return `Wipe ${target.store.title}?`;
    if (target.kind === "model") return `Wipe ${target.model.name}?`;
    return "Remove duplicates?";
  };

  const confirmLabel = () => {
    const target = confirm();
    if (!target) return "Confirm";
    if (target.kind === "store") return "Wipe store";
    if (target.kind === "model") return "Wipe";
    return `Reclaim ${fmtBytes(target.preview.reclaimable ?? null)}`;
  };

  const confirmDesc = () => {
    const target = confirm();
    if (!target) return "";
    if (target.kind === "store")
      return `This permanently deletes every model file in ${target.store.title} — ${fmtBytes(target.store.bytes)} reclaimed. This cannot be undone.`;
    if (target.kind === "model")
      return `This permanently deletes ${target.model.name} from ${target.store.title} — ${fmtBytes(target.model.bytes)} reclaimed. This cannot be undone.`;
    const dupes =
      target.preview.groups?.reduce((n, g) => n + (g.remove?.length ?? 0), 0) ?? 0;
    return `This permanently deletes ${fmtInt(dupes)} duplicate ${
      dupes === 1 ? "copy" : "copies"
    } marked "remove" across the stores — one copy of each model is kept. ${fmtBytes(
      target.preview.reclaimable ?? null
    )} reclaimed. This cannot be undone.`;
  };

  async function scanDedupe() {
    if (busy()) return;
    setBusy(true);
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
      setBusy(false);
    }
  }

  async function applyDedupe(preview: DedupePreview) {
    if (busy()) return;
    setBusy(true);
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
      setBusy(false);
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
          <div class="card-sub">with model files</div>
        </div>
        <div class="card">
          <div class="card-label">Models</div>
          <div class="card-value">{stores() === null ? "—" : fmtInt(modelCount())}</div>
          <div class="card-sub">entries across stores</div>
        </div>
      </div>

      <Show when={loading() && stores() === null}>
        <div class="panel">
          <div class="panel-row">
            <span class="grow muted">Scanning disk…</span>
          </div>
        </div>
      </Show>

      <Show when={loadError()} keyed>
        {(text: string) => <p class="notice error">{text}</p>}
      </Show>

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
                    {fmtBytes(disk().used_bytes)} of {fmtBytes(disk().capacity_bytes)}
                    <Show when={disk().persistent}>
                      {" "}
                      <span class="badge ok">persistent</span>
                    </Show>
                  </span>
                </div>
                <div class="panel-row">
                  <span class="grow">Cached KV</span>
                  <span class="num">
                    {fmtBytes(disk().kv_bytes)} in {fmtInt(disk().kv_blocks)} blocks
                    <Show when={disk().kv_pending_pages > 0}>
                      {" "}· {fmtInt(disk().kv_pending_pages)} pages pending
                    </Show>
                  </span>
                </div>
                <div class="panel-row">
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
                <div class="panel-row">
                  <span class="grow">Restores (SSD → RAM)</span>
                  <span class="num">
                    {fmtInt(disk().kv_restores)}
                    <Show when={disk().kv_restore_failures > 0}>
                      {" "}· {fmtInt(disk().kv_restore_failures)} failed
                    </Show>
                  </span>
                </div>
                <div class="panel-row">
                  <span class="grow">Write-behind</span>
                  <span class="num">
                    {fmtInt(disk().write_behind?.waiting)} waiting · {fmtInt(disk().write_behind?.durable)} durable
                    <Show when={(disk().write_behind?.refused ?? 0) > 0}>
                      {" "}· {fmtInt(disk().write_behind.refused)} refused
                    </Show>
                  </span>
                </div>
                <div class="panel-row">
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

      <section class="section">
        <h2>Duplicates</h2>
        <div class="dedupe-controls">
          <button
            class="btn"
            title="Find the same model stored in more than one place"
            disabled={busy() || loading()}
            onClick={scanDedupe}
          >
            {busy() && !dedupe() ? "Working…" : "Scan for duplicates"}
          </button>
          <label class="dedupe-variants muted">
            <input
              type="checkbox"
              title="Also group format variants (e.g. MLX vs GGUF) of the same model"
              checked={variants()}
              disabled={busy()}
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
                          <span class="badge">remove</span>
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
                      {fmtBytes(preview.reclaimable)} reclaimable
                    </span>
                    <button
                      class="btn danger"
                      title="Delete the copies marked remove — keeps one per model"
                      disabled={busy()}
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
        </div>
        <Show when={busy()}>
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
                    <span class="grow muted">{store.title} —</span>
                  </div>
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
                    disabled={busy() || store.locked}
                    title={
                      store.locked
                        ? "Stop the server before wiping the store it serves from"
                        : `Wipe all of ${store.title}`
                    }
                    onClick={() => setConfirm({ kind: "store", store })}
                  >
                    Wipe store
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
                      <span class="grow">{model.name}</span>
                      <span class="num">{fmtBytes(model.bytes)}</span>
                      <button
                        class="btn small danger"
                        disabled={busy() || store.locked}
                        title={
                          store.locked
                            ? "Store is in use by this server"
                            : `Wipe ${model.name}`
                        }
                        onClick={() => setConfirm({ kind: "model", store, model })}
                      >
                        Wipe
                      </button>
                    </div>
                  )}
                </For>
                <Show when={(store.models?.length ?? 0) === 0}>
                  <div class="panel-row">
                    <span class="grow muted">No model entries</span>
                  </div>
                </Show>
              </div>
            </Show>
          )}
        </For>
      </section>

      <Confirm
        open={confirm() !== null}
        title={confirmTitle()}
        confirmLabel={confirmLabel()}
        busy={busy()}
        onConfirm={runConfirmed}
        onClose={() => setConfirm(null)}
      >
        {confirmDesc()}
      </Confirm>
    </main>
  );
}
