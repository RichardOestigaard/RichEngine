// C-ABI shim for the CoreAI side of the predraft transport A/B
// (docs/ANE_EXPERIMENTS.md deep dive 6). Exposes just what the benchmark
// prices: specialize for the Neural Engine, then either the convenience
// run() path over preallocated NDArrays or encode() onto a Metal command
// queue with MTLBuffer-bound inputs/outputs.
import CoreAI
import Foundation
import Metal

private struct Slot {
  var model: AIModel?
  var function: InferenceFunction?
  var stream: ComputeStream?
  var hiddenArr: NDArray?
  var idArrs: [NDArray] = []
  var xbuf: (any MTLBuffer)?
  var idsbuf: (any MTLBuffer)?
  var ybuf: (any MTLBuffer)?
}
private var slot = Slot()

private func write(_ dst: inout NDArray, _ src: UnsafeRawPointer, _ bytes: Int) {
  var d = dst
  d.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
    ptr.copyMemory(from: src, byteCount: bytes)
  }
  dst = d
}

@_cdecl("coreai_shim_load")
public func coreai_shim_load(_ path: UnsafePointer<CChar>) -> Int32 {
  let url = URL(fileURLWithPath: String(cString: path))
  let sem = DispatchSemaphore(value: 0)
  var ok = false
  Task {
    do {
      let opts = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
      let model = try await AIModel(contentsOf: url, options: opts)
      if let f = try model.loadFunction(named: "main") {
        let dev = MTLCreateSystemDefaultDevice()!
        slot.model = model
        slot.function = f
        slot.stream = ComputeStream(commandQueue: dev.makeCommandQueue()!)
        // hidden [4,8,2048] fp16, three id vectors [4] fp32, proposals
        // [4,7,1] fp16 — the predraft contract.
        slot.hiddenArr = NDArray(shape: [4, 8, 2048], scalarType: .float16)
        slot.idArrs = (0..<3).map { _ in NDArray(shape: [4], scalarType: .float32) }
        slot.xbuf = dev.makeBuffer(length: 4 * 8 * 2048 * 2, options: .storageModeShared)!
        slot.idsbuf = dev.makeBuffer(length: 3 * 4 * 4, options: .storageModeShared)!
        slot.ybuf = dev.makeBuffer(length: 4 * 7 * 1 * 2, options: .storageModeShared)!
        ok = true
      }
    } catch { print("coreai_shim_load: \(error)") }
    sem.signal()
  }
  sem.wait()
  return ok ? 1 : 0
}

// Convenience path: copy caller data into preallocated NDArrays, run(),
// copy proposals back — the same in/out structure as CoreML predict().
@_cdecl("coreai_shim_eval_run")
public func coreai_shim_eval_run(_ hidden: UnsafeRawPointer,
                                 _ ids: UnsafeRawPointer,
                                 _ out: UnsafeMutableRawPointer) -> Int32 {
  guard let f = slot.function, var h = slot.hiddenArr else { return 0 }
  write(&h, hidden, 4 * 8 * 2048 * 2)
  for i in 0..<3 {
    var a = slot.idArrs[i]
    write(&a, ids + i * 16, 16)
    slot.idArrs[i] = a
  }
  slot.hiddenArr = h
  let sem = DispatchSemaphore(value: 0)
  var ok = false
  Task {
    do {
      var outputs = try await f.run(inputs: ["hidden": h, "anchor": slot.idArrs[0],
                                             "position": slot.idArrs[1],
                                             "retained": slot.idArrs[2]])
      if let v = outputs.remove("proposals"), let nd = v.ndArray {
        nd.rawView().withUnsafeBytes { src, _, _ in
          out.copyMemory(from: src, byteCount: 4 * 7 * 2)
        }
        ok = true
      }
    } catch { print("coreai_shim_eval_run: \(error)") }
    sem.signal()
  }
  sem.wait()
  return ok ? 1 : 0
}

// Zero-copy path: caller data lands in shared MTLBuffers, encode() binds
// them directly — the private-MIL surface transport's public equivalent.
@_cdecl("coreai_shim_eval_encode")
public func coreai_shim_eval_encode(_ hidden: UnsafeRawPointer,
                                    _ ids: UnsafeRawPointer,
                                    _ out: UnsafeMutableRawPointer) -> Int32 {
  guard let f = slot.function, let stream = slot.stream,
        let x = slot.xbuf, let ib = slot.idsbuf, let y = slot.ybuf else { return 0 }
  x.contents().copyMemory(from: hidden, byteCount: 4 * 8 * 2048 * 2)
  ib.contents().copyMemory(from: ids, byteCount: 3 * 4 * 4)
  do {
    var ins: [String: InferenceFunction.AsyncValue] = [:]
    ins["hidden"] = InferenceFunction.AsyncValue(unsafeBuffer: x, scalarType: .float16, shape: [4, 8, 2048])
    ins["anchor"] = InferenceFunction.AsyncValue(unsafeBuffer: ib, scalarType: .float32, shape: [4])
    ins["position"] = InferenceFunction.AsyncValue(unsafeBuffer: ib, byteOffset: 16, scalarType: .float32, shape: [4])
    ins["retained"] = InferenceFunction.AsyncValue(unsafeBuffer: ib, byteOffset: 32, scalarType: .float32, shape: [4])
    var outv = InferenceFunction.AsyncMutableViews()
    var yv = InferenceFunction.AsyncMutableValue(unsafeBuffer: y, scalarType: .float16, shape: [4, 7, 1])
    outv.insert(&yv, for: "proposals")
    _ = try f.encode(inputs: ins, outputViews: outv, to: stream)
  } catch { print("coreai_shim_eval_encode: \(error)"); return 0 }
  let sem = DispatchSemaphore(value: 0)
  Task { await stream.currentWorkCompleted(); sem.signal() }
  sem.wait()
  out.copyMemory(from: y.contents(), byteCount: 4 * 7 * 2)
  return 1
}
