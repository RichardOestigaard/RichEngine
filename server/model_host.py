"""The model stack the HTTP layer serves: load, swap and unload.

FrontendServer keeps one ModelHost; handlers read ``host.app`` — a Frontend
while a model is loaded, None otherwise. A load resolves an installed
selection, frees the loaded model before the new one's weights map (RAM
headroom is why a swap unloads first), then builds and publishes a fresh
stack. load() and unload() refuse while requests run, unless forced.
"""

import argparse
import math
import os
import threading
import time
from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer

from . import images as image_input
from . import runtime as engine_runtime
from . import serve_options
from .backend import NativeBackend
from .chat_templates import ChatTemplates
from .constraints import ConstraintFactory, validate_tokenizer
from .diagnostics import accent, dim, print_request, print_status
from .errors import APIError
from .frontend import Frontend
from .thinking import ThinkingCodec, load_thinking_key

NATIVE_START_TIMEOUT = 600.0
IDLE_WATCH_TICK = 30.0

# The states ModelHost.state cycles through.
HostState = Literal["unloaded", "loading", "loaded", "failed"]


def _native_command(args):
    """The native engine's argv: the assembly directory plus serve options."""
    command = [
        args.binary,
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

    def note_request(self):
        """A generation request arrived: it resets the idle-unload clock."""
        self.last_request_at = time.monotonic()

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
        selection = model_artifacts.Selection.of(
            self._models_root(),
            model_id,
            revision=revision,
            language_only=language_only,
            draft_model=draft_model,
        )
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
        contract = validate_tokenizer(tokenizer, str(tokenizer_path))
        chat_templates = ChatTemplates(tokenizer)
        print_status(f"Chat template · {chat_templates.describe()}")
        # Special tokens the output parser needs as text: the dialect's call
        # markup, and the think-close token where it is a special token.
        visible_token_ids = {contract.think_end_id, contract.think_open_id}
        for template in chat_templates.templates.values():
            dialect = template.dialect
            if dialect is None:
                continue
            for marker in dialect.structural:
                ids = tokenizer.encode(marker, add_special_tokens=False)
                if len(ids) == 1:
                    visible_token_ids.add(ids[0])
        visible_token_ids.discard(None)
        native_fields = vars(self.args).copy()
        native_fields["assembly_dir"] = str(assembly_dir)
        native_args = argparse.Namespace(**native_fields)
        # The canvas profile reaches the engine through the environment; an
        # explicit RICHENGINE_CANVAS_PROFILE wins over the flag, and both win
        # over autotune below. Only a non-default choice is exported — the
        # "paper" default would shadow a tuned profile forever.
        if self.args.canvas_profile != "paper":
            os.environ.setdefault("RICHENGINE_CANVAS_PROFILE", self.args.canvas_profile)
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
        runtime = engine_runtime.MultiplexedRuntime(
            _native_command(native_args),
            startup_timeout=NATIVE_START_TIMEOUT,
            pending_limit=self.args.queue_size,
            eager_start=False,
            env=autotune.tuned_environment(assembly_dir, binary=binary),
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
                thinking_codec=self.thinking_codec,
                served_model_names=self.args.served_model_name,
                announce_served_name=self.args.announce_served_name,
                default_reasoning_effort=self.args.default_reasoning_effort,
                vision=readiness.vision,
                contract=contract,
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
        self.record = record
        from install import autotune

        self.tuned_env = autotune.load_tuning(
            assembly_dir, binary=Path(self.args.binary)
        )
        self.state = "loaded"
        self.error = None
        self.last_request_at = time.monotonic()

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

    def unload(self, *, force=False):
        """Free the loaded model; its id, or None when none was loaded."""
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
