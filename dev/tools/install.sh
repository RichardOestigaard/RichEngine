#!/bin/sh
# RichEngine private-test installer for Apple Silicon Macs.
#
#   export RICHENGINE_TOKEN=hf_...
#   curl -qfsSL --config - <install.sh URL> <<EOF | RICHENGINE_REPO=owner/repo sh
#   header = "Authorization: Bearer $RICHENGINE_TOKEN"
#   EOF
#
# Downloads the pinned release archive from the private Hugging Face repo,
# verifies its SHA-256, unpacks it under ~/Library/Application Support/RichEngine/app,
# points the `current` link at it and writes a `richengine` command onto the PATH.
# Running it again upgrades in place; model weights and sessions are untouched.
#
#   RICHENGINE_TOKEN     read token from the invitation (required)
#   RICHENGINE_VERSION   install this version instead of the repo's `latest`
#   RICHENGINE_REPO      Hugging Face repo to install from (required)
#   RICHENGINE_BASE_URL  file server override, e.g. http://127.0.0.1:8123 for a local check
#   RICHENGINE_BIN_DIR   where to put the `richengine` command (default: Homebrew bin or ~/.local/bin)
set -eu

REPO=${RICHENGINE_REPO:-}
if [ -z "${RICHENGINE_BASE_URL:-}" ] && [ -z "$REPO" ]; then
  echo "set RICHENGINE_REPO to the Hugging Face repo to install from" >&2
  exit 1
fi
BASE_URL=${RICHENGINE_BASE_URL:-https://huggingface.co/$REPO/resolve/main}
TOKEN=${RICHENGINE_TOKEN:-}
APP="$HOME/Library/Application Support/RichEngine/app"
MARKER="RichEngine/app/current"

fail() { echo "richengine install: $*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || fail "RichEngine runs on Apple Silicon Macs only."
os=$(sw_vers -productVersion)
major=${os%%.*}
minor=${os#"$major"}
minor=${minor#.}
minor=${minor%%.*}
[ -n "$minor" ] || minor=0
[ "$major" -ge 27 ] 2>/dev/null \
    || fail "RichEngine requires macOS 27 or newer; this Mac runs $os."
command -v curl >/dev/null 2>&1 || fail "curl is required."
[ -n "$TOKEN" ] || fail "set RICHENGINE_TOKEN to the access token from your invitation."

dir=${RICHENGINE_BIN_DIR:-}
[ -n "$dir" ] || for candidate in /opt/homebrew/bin /usr/local/bin; do
    if [ -d "$candidate" ] && [ -w "$candidate" ]; then dir=$candidate; break; fi
done
[ -n "$dir" ] || dir="$HOME/.local/bin"
wrapper="$dir/richengine"
if [ -e "$wrapper" ] && ! grep -q "$MARKER" "$wrapper" 2>/dev/null; then
    fail "$wrapper exists and was not created by this installer; remove it first."
fi

fetch() {
    curl -q -fsSL --retry 3 --config "$work/curl.conf" -o "$2" "$BASE_URL/$1" \
        || fail "could not download $BASE_URL/$1 (expired or wrong token?)"
}

work=$(mktemp -d "${TMPDIR:-/tmp}/richengine-install.XXXXXX")
trap 'rm -rf "$work"' EXIT
case "$TOKEN" in *[!A-Za-z0-9_./~+=-]*) fail "invalid access token format.";; esac
(umask 077; printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" > "$work/curl.conf")

version=${RICHENGINE_VERSION:-}
if [ -z "$version" ]; then
    fetch latest "$work/latest"
    version=$(tr -d '[:space:]' < "$work/latest")
    [ -n "$version" ] || fail "the release index is empty."
fi
name="richengine-$version-arm64-macos26"

mkdir -p "$APP"
candidate="$APP/$name"
if [ -f "$candidate/release.json" ]; then
    echo "RichEngine $version is already downloaded."
else
    echo "Downloading RichEngine $version..."
    fetch "$name.tar.gz" "$work/$name.tar.gz"
    fetch "$name.tar.gz.sha256" "$work/$name.tar.gz.sha256"
    expected=$(cut -d' ' -f1 < "$work/$name.tar.gz.sha256")
    actual=$(shasum -a 256 "$work/$name.tar.gz" | cut -d' ' -f1)
    [ -n "$expected" ] && [ "$expected" = "$actual" ] || fail "checksum mismatch for $name.tar.gz."
    mkdir "$work/extract"
    tar -xzf "$work/$name.tar.gz" -C "$work/extract"
    [ -f "$work/extract/$name/release.json" ] || fail "unexpected archive layout."
    candidate="$work/extract/$name"
fi
# Validate before touching an installed version or waiting on its lifecycle lock.
if ! PYTHONDONTWRITEBYTECODE=1 "$candidate/python/bin/python3" -u \
        "$candidate/install/launcher.py" --help >/dev/null 2>&1; then
    fail "RichEngine $version fails 'richengine --help'; nothing was changed."
fi
cat > "$work/apply.sh" <<'INSTALL'
set -eu
APP=$1
name=$2
candidate=$3
wrapper=$4
dir=${wrapper%/*}
fail() { echo "richengine install: $*" >&2; exit 1; }
if [ "$candidate" != "$APP/$name" ] && [ ! -f "$APP/$name/release.json" ]; then
    rm -rf "$APP/$name"
    mv "$candidate" "$APP/$name"
fi
# Stage the wrapper before switching current; restore the old link if the
# final rename fails. Remove older versions only after both changes succeed.
mkdir -p "$dir"
cat > "$wrapper.tmp" <<WRAPPER || fail "could not write $wrapper; RichEngine $name was not installed."
#!/bin/sh
export PYTHONDONTWRITEBYTECODE=1
exec "$APP/current/python/bin/python3" -u "$APP/current/install/launcher.py" "\$@"
WRAPPER
chmod 0755 "$wrapper.tmp"
previous=$(readlink "$APP/current" 2>/dev/null || true)
ln -sfn "$APP/$name" "$APP/current"
if ! mv -f "$wrapper.tmp" "$wrapper"; then
    if [ -n "$previous" ]; then ln -sfn "$previous" "$APP/current"; else rm -f "$APP/current"; fi
    rm -f "$wrapper.tmp"
    fail "could not write $wrapper; RichEngine $name was not installed."
fi
for old in "$APP"/richengine-*-arm64-macos26; do
    [ -d "$old" ] && [ "$old" != "$APP/$name" ] && rm -rf "$old"
done
exit 0
INSTALL
# Use the validated bundled Python, so the installer needs no system Python.
# The descriptor survives exec and protects every switch, replacement and prune.
PYTHONDONTWRITEBYTECODE=1 "$candidate/python/bin/python3" - \
    "$APP/../runtime/serve.lock" "$work/apply.sh" "$APP" "$name" "$candidate" "$wrapper" <<'PYTHON'
import fcntl
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a+") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("richengine install: stop the running RichEngine server before upgrading; another installer may also hold the lock.")
    os.set_inheritable(lock.fileno(), True)
    os.execv("/bin/sh", ["/bin/sh", *sys.argv[2:]])
PYTHON

echo
echo "RichEngine $version installed: $wrapper"
echo "Private models require your own HF_TOKEN or 'hf auth login'."
case ":$PATH:" in
    *":$dir:"*) ;;
    *) echo "Add it to your PATH first:  export PATH=\"$dir:\$PATH\"" ;;
esac
echo "  richengine serve --model incoai/Qwen3.6-35B-A3B-RichEngine"
echo "  richengine claude|opencode|codex|hermes|pi   connect a coding agent to it"
if [ -f "$APP/current/install/completions/richengine.bash" ] && [ -f "$APP/current/install/completions/_richengine" ]; then
    echo "  Optional shell completion (Zsh needs compinit initialized):"
    echo '    Bash: source "$HOME/Library/Application Support/RichEngine/app/current/install/completions/richengine.bash"'
    echo '    Zsh:  source "$HOME/Library/Application Support/RichEngine/app/current/install/completions/_richengine"'
    if [ -f "$APP/current/install/completions/richengine.fish" ]; then
        echo '    Fish: source "$HOME/Library/Application Support/RichEngine/app/current/install/completions/richengine.fish"'
    fi
fi
echo "  Upgrade: run this installer again.  Uninstall: rm -rf \"$APP\" \"$wrapper\""
