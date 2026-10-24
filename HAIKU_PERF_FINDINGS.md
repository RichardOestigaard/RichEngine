# Performance Findings: Library Upgrade Candidates

Scope: Python server layer and C++/ObjC++ Metal runtime.

## Summary

- Metal GPU kernels do most of the work.
- Library swaps help Python edges only.
- C++ side already lean. Few gains.
- Profile before any swap.

## Python (ranked by payoff)

### 1. HTTP server

- Current: stdlib `ThreadingHTTPServer`, one thread per connection.
  - `server/server.py:19`
  - `server/server.py:2245` (`FrontendServer`)
- Options: uvicorn, granian, hypercorn (asyncio).
- Caveat: custom admission and connection logic in `server.py` must port.

### 2. JSON codec

- Current: stdlib `json`.
  - `server/json_codec.py`
- Options: orjson, msgspec.
- Caveats:
  - Keep U+2028/2029 and surrogate escaping.
  - orjson has no `parse_float` hook.

### 3. Tokenizer wrapper

- Current: `transformers` `PreTrainedTokenizerFast` calls on hot paths.
  - `server/tokenization.py:79`
  - `server/backend.py:230-296`
  - `server/judgments.py`
- Option: raw `tokenizers.Tokenizer`. Code already uses it for `DecodeStream`.
- Bonus: drops heavy `transformers` import at `server/model_host.py:18`. Faster startup.
- Caveat: `apply_chat_template` still needs `transformers` or a Jinja port.

### 4. Free-threaded Python

- Current: heavy threading. `ThreadPoolExecutor` at `server/runtime.py:21`.
- Option: Python 3.13t or 3.14t (no GIL).
- Caveat: verify free-threaded wheels for `tokenizers`, `numpy`, `regex`, `Pillow`.

### 5. JSON schema validation

- Current: `jsonschema`, pure Python. Already cached.
  - `server/schema_validation.py`
- Option: Rust or `fastjsonschema`.
- Only worth it if profiling shows cost.

### 6. Offline quantization

- Current: `numpy` in `install/pack.py:117-148`.
- Option: MLX (Metal) or Accelerate via ctypes.
- Scope: install time only. No runtime impact.

## C++ / ObjC++

### Already lean (skip)

- CRC32C uses ARM intrinsics. `runtime/model/Crc32c.hpp`
- Hand-written JSON quoting only. `runtime/engine/wire/Json.hpp`
- Binary frame protocol. `runtime/engine/wire/Protocol.cpp`
- Do not add nlohmann, simdjson, or rapidjson.

### Hash maps

- Current: `std::unordered_map`, `unordered_set`, `std::map` at:
  - `runtime/model/NgramIndex.hpp:26`
  - `runtime/engine/Engine.cpp:556`
  - `runtime/engine/cache/KvCache.cpp:346`
  - `runtime/engine/cache/Cache.cpp:53`
- Option: `absl::flat_hash_map` or `ankerl::unordered_dense`.
- Only if profiling flags them.

### Allocator

- Option: mimalloc.
- Modest gain. Measure first.

### Compiler flags

- Current: `-O3` at `Makefile:113`.
- Try: `-flto=thin`, `-mcpu=apple-m5`. Tuning data comes from M5 Pro.
- Avoid `-ffast-math`. Breaks numerics.

### Accelerate (vDSP/BLAS)

- No hot CPU math loop found.
- Compute runs on Metal and ANE already.

## Next Steps

1. Profile Python: `py-spy` or Instruments.
2. Profile C++: Instruments Time Profiler.
3. Start with HTTP server and tokenizer wrapper.
4. Benchmark each swap against baseline.
