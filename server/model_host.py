"""The model stack the HTTP layer serves: load, swap and unload.

FrontendServer keeps one ModelHost; handlers read ``host.app`` — a Frontend
while a model is loaded, None otherwise. A load resolves an installed
selection, frees the loaded model before the new one's weights map (RAM
headroom is why a swap unloads first), then builds and publishes a fresh
stack. load() and unload() refuse while requests run, unless forced.
"""

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer

from . import images as image_input
from . import runtime as engine_runtime
from . import serve_options, serving
from .backend import NativeBackend
from .chat_templates import ChatTemplates
from .constraints import ConstraintFactory, validate_tokenizer
from .diagnostics import accent, dim, print_request, print_status
from .errors import APIError
from .frontend import Frontend
from .thinking import ThinkingCodec, load_thinking_key
from .tool_schema import dialect_named

NATIVE_START_TIMEOUT = 600.0
IDLE_WATCH_TICK = 30.0

# The states ModelHost.state cycles through.
HostState = Literal["unloaded", "loading", "loaded", "failed"]


def _engine_binary(binary: str, assembly_dir: Path) -> str:
    """The shipped binary: it carries both verify geometries now.

    Older builds kept a block-8 variant in a sibling build8/ directory
    (packaged: engine8/) for drafts trained at eight rows; the unified
    richengine's dispatch picks the variant itself from the model's draft
    geometry. Keep returning one for trees still built that way.
    """
    try:
        draft = json.loads((assembly_dir / "draft" / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return binary
    block_size = draft.get("dflash_config", {}).get("block_size")
    if not block_size or block_size > 8:
        return binary
    variant = (
        Path(binary).parent.parent / (Path(binary).parent.name + "8") / "richengine"
    )
    return str(variant) if variant.is_file() else binary


def _native_command(args):
    """The native engine's argv: the assembly directory plus serve options."""
    command = [
        _engine_binary(args.binary, Path(args.assembly_dir)),
        "serve-native",
        str(args.assembly_dir),
        "auto" if args.max_context is None else str(args.max_context),
        "auto" if args.max_memory is None else str(args.max_memory),
    ]
    if args.max_cache_disk:
        command.append(str(args.max_cache_disk))
    if args.persistent_cache:
        command.extend(
            ("--cache-dir", str(args.cache_dir or serve_options.DEFAULT_CACHE_DIR))
        )
    if args.kv_format != "int4":
        command.extend(("--kv-format", args.kv_format))
    if args.idle_release is not None:
        command.extend(
            ("--idle-release", serve_options.idle_release_text(args.idle_release))
        )
    if args.disable_ane:
        command.extend(("--ane", "off"))
    if args.decode_share is not None:
        command.extend(("--decode-share", str(args.decode_share)))
    if args.moe_union:
        command.extend(("--moe-union", str(args.moe_union)))
    if args.max_image_pixels != image_input.MAX_PIXELS:
        command.extend(
            ("--max-image-patches", str(image_input.max_patches(args.max_image_pixels)))
        )
    if args.allow_idle_sleep:
        command.extend(("--idle-sleep", "allow"))
    weight_layout = getattr(args, "weight_layout", "auto")
    if weight_layout != "auto":
        command.extend(("--weight-layout", weight_layout))
    return command


class ModelHost:
    """Owns the loaded tokenizer/runtime/backend/Frontend under one lock."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.app = None
        self.backend = None
        self.runtime = None
        self.assembly_dir = None
        # assembly.hold's record fd: while open, collection keeps the
        # assembly this model serves from.
        self.record = None
        # The tuned env the running engine spawned with; a tune finished
        # since only reaches it through a reload.
        self.tuned_env = None
        # A HostState: unloaded | loading | loaded | failed.
        self.state: HostState = "unloaded"
        self.error = None
        self.closed = False
        self.last_request_at = time.monotonic()
        self.thinking_codec = ThinkingCodec(load_thinking_key())
        # The effective weight layout of the loaded model, for /status.
        self.weight_layout = None
        self._wakeup = threading.Event()
        self._watchdog = None
        if args.unload_idle is not None and math.isfinite(args.unload_idle):
            self._watchdog = threading.Thread(
                target=self._idle_watch,
                name="richengine-idle-unload",
                daemon=True,
            )
            self._watchdog.start()

    # -- state ---------------------------------------------------------

    def note_request(self, app=None):
        """A generation request arrived: it resets the idle-unload clock.
        The app argument keeps the pool's call shape; one host serves it."""
        self.last_request_at = time.monotonic()

    # -- the pool protocol: a lone host answers for its one model ---------

    def apps(self):
        return [self.app] if self.app is not None else []

    def models(self):
        return list(self.app.model_names) if self.app is not None else []

    def host_for_app(self, app):
        return self if self.app is app else None

    def decision_host(self):
        if self.app is not None and self.app.decision_pathway:
            return self
        return None

    def app_for(self, model, *, prefer_decision=False):
        """Single-host routing: a name resolves here or nowhere."""
        if model is None or (
            self.app is not None and model in self.app.model_names
        ):
            return self.app
        return None

    def _require_idle_locked(self, force):
        """Refuse a swap or unload while the engine still serves requests."""
        if force or self.backend is None:
            return
        with self.backend.lock:
            busy = len(self.backend.active)
        if busy:
            raise APIError(
                409,
                f"{busy} request{'s' if busy != 1 else ''} in flight; "
                'retry when they finish, or pass "force": true',
                "engine_busy",
            )

    # -- resolution -----------------------------------------------------

    def _models_root(self):
        if self.args.models_dir is not None:
            return Path(self.args.models_dir)
        try:
            # The launcher passes its cache root; a bare `python -m
            # server.server` resolves it through the install package.
            from install import paths
        except ImportError:
            raise APIError(
                400,
                "the models directory is unknown; restart with --models-dir",
                "invalid_request_error",
            ) from None
        return paths.MODELS

    def _hold(self, model_id, *, revision=None, draft_model=None, language_only=False):
        """The installed selection's held assembly directory and record fd."""
        from install import assembly
        from install import models as model_artifacts

        try:
            model_artifacts.parse_model_id(model_id)
        except argparse.ArgumentTypeError as error:
            raise APIError(400, str(error)) from None
        try:
            selection = model_artifacts.Selection.of(
                self._models_root(),
                model_id,
                revision=revision,
                language_only=language_only,
                draft_model=draft_model,
            )
        except model_artifacts.ModelError as error:
            raise APIError(400, str(error)) from None
        if model_artifacts.installation_kind(selection.link) is None:
            raise APIError(
                404,
                f"{model_id} is not installed; install it with "
                f"'richengine serve --model {model_id}' first",
                "model_not_found",
            )
        try:
            return assembly.hold(selection.link, selection.models_root)
        except model_artifacts.ModelError as error:
            raise APIError(400, str(error)) from None

    # -- build ----------------------------------------------------------

    def _build(self, assembly_dir, tokenizer_path, model_id):
        """Tokenizer, templates, runtime, backend and Frontend for a held
        assembly, validated against the engine's Ready. On any failure the
        runtime is stopped; nothing is published."""
        print_status(f"Loading · {accent(model_id)}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        # The assembly's serving contract: the spellings and defaults this
        # model's record declares (or its family file does), per model, not
        # per server.
        serving_contract = serving.load(assembly_dir)
        contract = validate_tokenizer(tokenizer, str(tokenizer_path), serving_contract)
        chat_templates = ChatTemplates(
            tokenizer,
            allow_missing=bool(
                serving_contract.decision or serving_contract.encoder
            ),
        )
        print_status(f"Chat template · {chat_templates.describe()}")
        # Special tokens the output parser needs as text: the dialect's call
        # markup, and the think-close token where it is a special token.
        visible_token_ids = {contract.think_end_id, contract.think_open_id}
        dialects = {
            template.dialect
            for template in chat_templates.templates.values()
            if template.dialect is not None
        }
        # The contract's dialect counts too: the probe may render a template
        # whose call markup it cannot classify while the family file names it.
        declared = dialect_named(serving_contract.tool_dialect)
        if declared is not None:
            dialects.add(declared)
        for dialect in dialects:
            for marker in dialect.structural:
                ids = tokenizer.encode(marker, add_special_tokens=False)
                if len(ids) == 1:
                    visible_token_ids.add(ids[0])
        visible_token_ids.discard(None)
        native_fields = vars(self.args).copy()
        native_fields["assembly_dir"] = str(assembly_dir)
        native_args = argparse.Namespace(**native_fields)
        # Autotune's per-chip knob winners (install/autotune.py's sweep) reach
        # the engine as environment at spawn; any variable the user set wins.
        # A record made on another engine build is stale and ignored.
        from install import autotune

        binary = Path(self.args.binary)
        note = autotune.tuning_note(assembly_dir, binary=binary)
        if note:
            print_status(f"Autotune · {note}")
        tuned = autotune.load_tuning(assembly_dir, binary=binary)
        if tuned:
            overridden = sorted(name for name in tuned if name in os.environ)
            applied = sorted(
                f"{name.removeprefix('RICHENGINE_')}={tuned[name]}"
                for name in tuned
                if name not in os.environ
            )
            print_status(
                "Autotune · "
                + (", ".join(applied) if applied else "all overridden")
                + (f" · your env keeps {', '.join(overridden)}" if overridden else "")
            )
        # The canvas profile reaches the engine through the environment; an
        # explicit RICHENGINE_CANVAS_PROFILE wins over the flag, and both win
        # over the tuned value below. Only a non-default choice is exported —
        # the "paper" default would shadow a tuned profile forever — and it
        # goes to this engine's spawn environment only, so one model's
        # profile never persists into the next model's.
        canvas_profile = self.args.canvas_profile
        tuned_env = autotune.tuned_environment(assembly_dir, binary=binary)
        if canvas_profile != "paper":
            env = tuned_env if tuned_env is not None else dict(os.environ)
            env["RICHENGINE_CANVAS_PROFILE"] = os.environ.get(
                "RICHENGINE_CANVAS_PROFILE", canvas_profile
            )
            tuned_env = env
        runtime = engine_runtime.MultiplexedRuntime(
            _native_command(native_args),
            startup_timeout=NATIVE_START_TIMEOUT,
            pending_limit=self.args.queue_size,
            eager_start=False,
            env=tuned_env,
        )
        # The packed manifest names diffusion targets (packed-only families
        # install as packages; an assembly has no manifest); their canvas
        # bursts need burst-aware throughput in the request metrics.
        from install import layout
        from install import models as model_artifacts

        try:
            manifest = (
                model_artifacts.read_json(assembly_dir / layout.PACKAGE_MANIFEST)
                if model_artifacts.installation_kind(assembly_dir)
                == model_artifacts.PACKAGE
                else {}
            )
            diffusion = isinstance(manifest.get("diffusion"), dict)
        except (model_artifacts.ModelError, OSError):
            diffusion = False
        backend = NativeBackend(
            runtime,
            tokenizer,
            request_logger=print_request,
            think_end_id=contract.think_end_id,
            visible_token_ids=visible_token_ids,
            tool_call_open_id=contract.tool_call_open_id,
            diffusion=diffusion,
        )
        try:
            if not runtime.wait_ready():
                raise engine_runtime.EngineUnhealthy(
                    "native runtime did not become ready"
                )
            readiness = runtime.readiness
            if (
                readiness is None
                or not 1
                <= readiness.max_context_tokens
                <= serve_options.MAX_CONTEXT_TOKENS
                or (
                    self.args.max_context is not None
                    and readiness.max_context_tokens != self.args.max_context
                )
            ):
                raise engine_runtime.EngineUnhealthy(
                    "native runtime reported an invalid context window"
                )
            effective_context = readiness.max_context_tokens
            constraint_factory = ConstraintFactory(tokenizer, contract)
            app = Frontend(
                tokenizer,
                backend,
                model_id,
                effective_context,
                math.inf
                if self.args.request_timeout is None
                else self.args.request_timeout,
                readiness.max_concurrent_requests,
                constraint_factory=constraint_factory,
                chat_templates=chat_templates,
                max_image_pixels=self.args.max_image_pixels,
                thinking_codec=self.thinking_codec.scoped(model_id),
                served_model_names=self.args.served_model_name,
                announce_served_name=self.args.announce_served_name,
                default_reasoning_effort=self.args.default_reasoning_effort,
                vision=readiness.vision,
                contract=contract,
                serving=serving_contract,
                shared_prefix_states=self.args.shared_prefix_state,
            )
        except BaseException:
            backend.close()
            raise
        return runtime, backend, app, effective_context, readiness.vision

    def _publish_locked(self, stack, assembly_dir, record):
        runtime, backend, app, _, _ = stack
        self.runtime = runtime
        self.backend = backend
        self.app = app
        self.assembly_dir = assembly_dir
        self.weight_layout = self._weight_layout(assembly_dir)
        self.record = record
        from install import autotune

        self.tuned_env = autotune.load_tuning(
            assembly_dir, binary=Path(self.args.binary)
        )
        self.state = "loaded"
        self.error = None
        self.last_request_at = time.monotonic()

    def _weight_layout(self, assembly_dir):
        """Fallback weight layout for an engine that reports no
        weights.layout in /status: it predates block MLX weights, so an
        mlx-affine target ran affine; GGUF targets always ran block and a
        pack.py target its own packed form."""
        from install import layout
        from install import models as model_artifacts

        try:
            record = model_artifacts.read_json(assembly_dir / layout.ASSEMBLY_RECORD)
            target_format = record.get("target_format")
        except (model_artifacts.ModelError, OSError):
            target_format = (
                "gguf"
                if any((assembly_dir / layout.TARGET).glob("*.gguf"))
                else None
            )
        if target_format == "gguf":
            return "block"
        if target_format == "mlx-affine":
            return "affine"
        return "packed"

    def tuning_requires_reload(self):
        """True when tuning.json now states winners this engine never saw:
        the runtime keeps the env it spawned with, so a tune finished while
        the model stayed loaded applies only on the next load."""
        if self.state != "loaded" or self.assembly_dir is None:
            return False
        from install import autotune

        return autotune.load_tuning(
            self.assembly_dir, binary=Path(self.args.binary)
        ) != (self.tuned_env or {})

    def _unload_locked(self):
        """Release the stack. Active requests end as server-shutdown errors."""
        backend = self.backend
        record = self.record
        self.app = None
        self.backend = None
        self.runtime = None
        self.assembly_dir = None
        self.weight_layout = None
        self.record = None
        self.tuned_env = None
        self.state = "unloaded"
        if backend is not None:
            backend.close()
        if record is not None:
            record.close()

    # -- API ----------------------------------------------------------------

    def load(
        self,
        model_id,
        *,
        force=False,
        revision=None,
        draft_model=None,
        language_only=False,
    ):
        """Swap to an installed model. The old engine is freed before the new
        one's weights map; a busy engine refuses unless forced."""
        with self.lock:
            if self.closed:
                raise APIError(503, "server is shutting down", "server_shutdown")
            if self.state == "loading":
                raise APIError(
                    409, "a model load is already in progress", "engine_busy"
                )
            self._require_idle_locked(force)
        record = None
        try:
            # The filesystem walk runs outside the lock; it is cheap.
            root, record = self._hold(
                model_id,
                revision=revision,
                draft_model=draft_model,
                language_only=language_only,
            )
            with self.lock:
                if self.closed:
                    raise APIError(503, "server is shutting down", "server_shutdown")
                # A request may have arrived while the selection resolved.
                self._require_idle_locked(force)
                self._unload_locked()
                self.state = "loading"
                self.error = None
            try:
                stack = self._build(root, root / "tokenizer", model_id)
            except Exception as error:
                with self.lock:
                    self.state = "unloaded"
                    self.error = str(error)
                raise APIError(500, f"model failed to load: {error}") from error
            with self.lock:
                self._publish_locked(stack, root, record)
                record = None
            app = self.app
            context = (
                f"{app.max_context // 1024}K"
                if app.max_context % 1024 == 0
                else f"{app.max_context:,}"
            )
            mode = "" if app.vision else " · language only"
            print_status(
                f"Ready · {accent(model_id)} · {dim(f'context {context}{mode}')}"
            )
            return app
        finally:
            if record is not None:
                record.close()

    def load_initial(self):
        """The model the command line named: the launcher already holds its
        assembly, so no resolution or hold is needed."""
        if self.args.model is None:
            return
        with self.lock:
            self.state = "loading"
        try:
            stack = self._build(
                Path(self.args.assembly_dir),
                Path(self.args.tokenizer),
                self.args.model,
            )
        except Exception as error:
            with self.lock:
                self.state = "failed"
                self.error = str(error)
            raise
        with self.lock:
            self._publish_locked(stack, Path(self.args.assembly_dir), None)
        # The Ready line — with its address — is main()'s to print after the
        # HTTP listener activates.

    def unload(self, model_id=None, *, force=False):
        """Free the loaded model; its id, or None when none was loaded. A
        named model must be this host's — the pool's uniform call shape."""
        if (
            model_id is not None
            and self.app is not None
            and model_id not in self.app.model_names
        ):
            raise APIError(
                404, f"model {model_id} is not loaded", "model_not_found"
            )
        with self.lock:
            if self.closed:
                raise APIError(503, "server is shutting down", "server_shutdown")
            if self.state == "loading":
                raise APIError(409, "a model load is in progress", "engine_busy")
            self._require_idle_locked(force)
            model = None if self.app is None else self.app.model
            if self.backend is not None or self.app is not None:
                print_status(
                    f"Unloading · {dim(model or 'model')} · "
                    f"{dim('releasing engine resources')}"
                )
            self._unload_locked()
        return model

    # -- lifecycle -----------------------------------------------------------

    def _idle_watch(self):
        """--unload-idle: unload the model after this long without a request."""
        idle_seconds = self.args.unload_idle
        while not self._wakeup.wait(min(IDLE_WATCH_TICK, idle_seconds)):
            with self.lock:
                if (
                    self.closed
                    or self.app is None
                    or self.state != "loaded"
                    or time.monotonic() - self.last_request_at < idle_seconds
                    or (self.backend is not None and bool(self.backend.active))
                ):
                    continue
                model = self.app.model
            print_status(
                f"Idle unload · {accent(model)} after {idle_seconds:g}s without a request"
            )
            with self.lock:
                self._unload_locked()

    def close(self):
        """Server shutdown: stop the watchdog and release the stack."""
        with self.lock:
            self.closed = True
        self._wakeup.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)
        with self.lock:
            self._unload_locked()


class HostPool:
    """Several resident ModelHost stacks behind one server.

    The pool owns the resident set — one ModelHost per resident model, each
    with its own native subprocess, KV pool and lifecycle — and routes by
    model name. It also stands in for the host where the server's compat
    surface only ever means "the default resident": ``app``, ``state``,
    ``error``, ``assembly_dir``, ``weight_layout``,
    ``tuning_requires_reload()`` and ``close()`` all read the first host's.
    ``max_residents`` bounds the set; at 1, load() reuses the single host
    and its in-place swap is exactly the old single-model behavior.
    """

    def __init__(self, args, max_residents=1, default_host=None):
        self.args = args
        self.max_residents = max(1, max_residents)
        self.hosts = [default_host if default_host is not None else ModelHost(args)]
        # Membership (the resident list) is rare to change; each host's own
        # lock still orders its stack.
        self.lock = threading.RLock()

    # -- resident set ---------------------------------------------------

    @property
    def default(self):
        """The primary resident: the first host, or None of an empty pool."""
        return self.hosts[0] if self.hosts else None

    def _primary(self):
        """The resident the pool answers as: the default host while it
        serves, else the first loaded resident — an empty default slot
        must not strand the residents that did load."""
        if self.default is not None and self.default.app is not None:
            return self.default
        return next(
            (host for host in self.hosts if host.app is not None),
            self.default,
        )

    def apps(self):
        """Every resident's live Frontend, default first."""
        return [host.app for host in self.hosts if host.app is not None]

    def models(self):
        """Every name any resident serves under, default first."""
        return [name for app in self.apps() for name in app.model_names]

    def host_for_app(self, app):
        """The host owning app, or None."""
        for host in self.hosts:
            if host.app is app:
                return host
        return None

    def decision_host(self):
        """The resident running a decision pathway, if any."""
        for host in self.hosts:
            if host.app is not None and host.app.decision_pathway:
                return host
        return None

    def app_for(self, model, *, prefer_decision=False):
        """The app serving model across residents; None routes to the
        decision host for decision endpoints, else the primary."""
        if model is None:
            if prefer_decision:
                host = self.decision_host()
                if host is not None:
                    return host.app
            return self.app
        for host in self.hosts:
            if host.app is not None and model in host.app.model_names:
                return host.app
        return None

    def note_request(self, app=None):
        """A generation request arrived; reset the serving host's clock."""
        host = self.host_for_app(app) if app is not None else self._primary()
        if host is not None:
            host.note_request()

    # -- compat: the primary resident's surface --------------------------

    @property
    def app(self):
        host = self._primary()
        return host.app if host is not None else None

    @property
    def state(self):
        host = self._primary()
        return host.state if host is not None else "unloaded"

    @property
    def error(self):
        host = self._primary()
        return host.error if host is not None else None

    @property
    def backend(self):
        host = self._primary()
        return host.backend if host is not None else None

    @property
    def runtime(self):
        host = self._primary()
        return host.runtime if host is not None else None

    @property
    def assembly_dir(self):
        host = self._primary()
        return host.assembly_dir if host is not None else None

    @property
    def weight_layout(self):
        host = self._primary()
        return host.weight_layout if host is not None else None

    def tuning_requires_reload(self):
        host = self._primary()
        return host is not None and host.tuning_requires_reload()

    def _models_root(self):
        return self.default._models_root()

    def residents_status(self):
        """Per-resident /status rows."""
        rows = {}
        for host in self.hosts:
            if host.app is None:
                continue
            # Each resident's engine answers its own status, as the
            # default's does for the top-level row; memory_actual rides it.
            memory = host.app.status().get("memory_actual") or {}
            rows[host.app.model] = {
                "state": host.state,
                "model_state": host.state,
                "error": host.error,
                "context_length": host.app.max_context,
                "vision": host.app.vision,
                "decision": host.app.decision_pathway,
                "input_modalities": host.app.input_modalities,
                "memory_bytes": memory.get("current_bytes"),
                "memory_peak_bytes": memory.get("peak_bytes"),
            }
        return rows

    def pool_status(self):
        """/status's capacity summary: hosts in use plus the loads left
        before load() refuses with its 409."""
        return {
            "max_residents": self.max_residents,
            "slots": len(self.hosts),
            "free": (
                sum(host.app is None for host in self.hosts)
                + self.max_residents
                - len(self.hosts)
            ),
        }

    def residents_ready(self):
        """/ready's resident counts by host state."""
        counts = {
            "total": len(self.hosts),
            "loaded": 0,
            "loading": 0,
            "unloaded": 0,
            "failed": 0,
        }
        for host in self.hosts:
            counts[host.state] = counts.get(host.state, 0) + 1
        return counts

    # -- lifecycle --------------------------------------------------------

    def load_initial(self):
        """The command line's models: the primary via its held assembly,
        every --extra-model through the installed-selection path."""
        self.default.load_initial()
        for model_id in getattr(self.args, "extra_model", None) or ():
            self.load(model_id)

    def load(
        self,
        model_id,
        *,
        force=False,
        revision=None,
        draft_model=None,
        language_only=False,
    ):
        """Load model_id as a resident: a free slot, a new one under
        max_residents, or — at one resident — the host's own swap."""
        with self.lock:
            host = None
            for candidate in self.hosts:
                if (
                    candidate.app is not None
                    and model_id in candidate.app.model_names
                ):
                    host = candidate
                    break
            if host is None:
                free = next(
                    (candidate for candidate in self.hosts if candidate.app is None),
                    None,
                )
                if free is not None:
                    host = free
                elif len(self.hosts) < self.max_residents:
                    host = ModelHost(self.args)
                    self.hosts.append(host)
                elif self.max_residents == 1:
                    host = self.default
                else:
                    raise APIError(
                        409,
                        f"resident limit {self.max_residents} reached; "
                        "unload a model first",
                        "engine_busy",
                    )
        return host.load(
            model_id,
            force=force,
            revision=revision,
            draft_model=draft_model,
            language_only=language_only,
        )

    def unload(self, model_id=None, *, force=False):
        """Unload one resident's stack; the slot stays for a later load."""
        with self.lock:
            if model_id is None:
                host = self._primary()
            else:
                host = next(
                    (
                        candidate
                        for candidate in self.hosts
                        if candidate.app is not None
                        and model_id in candidate.app.model_names
                    ),
                    None,
                )
            if host is None:
                raise APIError(
                    404, f"model {model_id} is not loaded", "model_not_found"
                )
        return host.unload(force=force)

    def close(self):
        """Server shutdown: release every resident."""
        for host in list(self.hosts):
            host.close()
