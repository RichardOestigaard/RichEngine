// librichengine_coreai.dylib — the C ABI of ops/CoreAiBridge.h over the
// Swift-only CoreAI framework: one loaded InferenceFunction of a dynamic-
// weight dense-FFN .aimodel (dev/tools/gen_coreai_ffn.py), evaluated by
// encode() onto a ComputeStream of its own MTLCommandQueue. Each job waits
// for the caller's MTLSharedEvent to reach `wait` and raises it to
// `signal`, the same protocol ane::Handoff drives, so the evaluation
// orders inside the caller's prefill command exactly as the private
// program's does.
import CoreAI
import Foundation
import Metal

private final class Slot {
  let model: AIModel
  let function: InferenceFunction
  let stream: ComputeStream
  let queue: any MTLCommandQueue
  let event: any MTLSharedEvent
  let hidden: Int
  let ane: Int
  // The asset's weight contract, read off its descriptors at open: packed
  // means int4 weights (half the q bytes); blockwise means one fp16 scale
  // per 32 inputs (gate_s [ane, hidden/32]) rather than per row [ane].
  let packed: Bool
  let blockwise: Bool
  // One evaluation on the shared x/y buffers at a time. A semaphore, not a
  // lock: the wait is released from whichever thread the task resumed on.
  let jobs = DispatchSemaphore(value: 1)

  init(model: AIModel, function: InferenceFunction, queue: any MTLCommandQueue, event: any MTLSharedEvent,
       hidden: Int, ane: Int, packed: Bool, blockwise: Bool) {
    self.model = model
    self.function = function
    self.queue = queue
    self.stream = ComputeStream(commandQueue: queue)
    self.event = event
    self.hidden = hidden
    self.ane = ane
    self.packed = packed
    self.blockwise = blockwise
  }

  var qType: NDArray.ScalarType { packed ? .int4 : .int8 }
  // Bytes of a [rows, width] weight input and of its scale input.
  func qBytes(_ rows: Int, _ width: Int) -> Int { rows * width / (packed ? 2 : 1) }
  func sShape(_ rows: Int, _ width: Int) -> [Int] { blockwise ? [rows, width / 32] : [rows] }
  func sBytes(_ rows: Int, _ width: Int) -> Int { rows * (blockwise ? width / 32 : 1) * 2 }
}

private var lastError: NSString?

private func fail(_ message: String, _ error: UnsafeMutablePointer<UnsafePointer<CChar>?>?) -> UnsafeMutableRawPointer? {
  lastError = message as NSString
  error?.pointee = UnsafePointer(lastError!.utf8String)
  return nil
}

private func warmUp(_ slot: Slot, on device: any MTLDevice) async throws {
  let h = slot.hidden, a = slot.ane
  func buffer(_ bytes: Int) -> any MTLBuffer {
    device.makeBuffer(length: bytes, options: .storageModeShared)!
  }
  // A cold function specializes per activation shape, which can outlast the
  // caller's job bound: burn the row counts a job can take while load
  // latency is expected.
  for rows in [1, 512, 2048] {
    let inputs: [String: InferenceFunction.AsyncValue] = [
      "x": .init(unsafeBuffer: buffer(rows * h * 2), scalarType: .float16, shape: [rows, h]),
      "gate_q": .init(unsafeBuffer: buffer(slot.qBytes(a, h)), scalarType: slot.qType, shape: [a, h]),
      "gate_s": .init(unsafeBuffer: buffer(slot.sBytes(a, h)), scalarType: .float16, shape: slot.sShape(a, h)),
      "up_q": .init(unsafeBuffer: buffer(slot.qBytes(a, h)), scalarType: slot.qType, shape: [a, h]),
      "up_s": .init(unsafeBuffer: buffer(slot.sBytes(a, h)), scalarType: .float16, shape: slot.sShape(a, h)),
      "down_q": .init(unsafeBuffer: buffer(slot.qBytes(h, a)), scalarType: slot.qType, shape: [h, a]),
      "down_s": .init(unsafeBuffer: buffer(slot.sBytes(h, a)), scalarType: .float16, shape: slot.sShape(h, a)),
    ]
    var yv = InferenceFunction.AsyncMutableValue(unsafeBuffer: buffer(rows * h * 2),
                                                 scalarType: .float16, shape: [rows, h])
    var outputs = InferenceFunction.AsyncMutableViews()
    outputs.insert(&yv, for: "y")
    _ = try slot.function.encode(inputs: inputs, outputViews: outputs, to: slot.stream)
    _ = try await yv.ndArray
  }
}

private func shape(_ f: InferenceFunction, _ name: String) -> [Int]? {
  guard let d = f.descriptor.inputDescriptor(of: name), case .ndArray(let nd) = d else { return nil }
  return nd.shape
}

@_cdecl("coreai_ffn_open")
public func coreai_ffn_open(_ path: UnsafePointer<CChar>, _ name: UnsafePointer<CChar>,
                            _ device: UnsafeMutableRawPointer, _ event: UnsafeMutableRawPointer,
                            _ hidden: UInt64, _ ane: UInt64,
                            _ error: UnsafeMutablePointer<UnsafePointer<CChar>?>?) -> UnsafeMutableRawPointer? {
  let device = Unmanaged<AnyObject>.fromOpaque(device).takeUnretainedValue() as! any MTLDevice
  let event = Unmanaged<AnyObject>.fromOpaque(event).takeUnretainedValue() as! any MTLSharedEvent
  let url = URL(fileURLWithPath: String(cString: path))
  let functionName = String(cString: name)
  let sem = DispatchSemaphore(value: 0)
  var result: UnsafeMutableRawPointer?
  var failure: String?
  Task {
    do {
      let model = try await AIModel(contentsOf: url,
                                    options: SpecializationOptions(preferredComputeUnitKind: .neuralEngine))
      guard let f = try model.loadFunction(named: functionName) else {
        failure = "the asset has no function \(functionName)"
        sem.signal()
        return
      }
      // The function's contract, inferred from the asset: x/y fp16
      // [rows, hidden]; gate/up [ane, hidden] and down [hidden, ane]
      // int8 or packed int4; their fp16 scales per output channel
      // (per-channel) or per 32-input block (blockwise).
      let h = Int(hidden), a = Int(ane)
      var packed = false
      if let d = f.descriptor.inputDescriptor(of: "gate_q"), case .ndArray(let nd) = d {
        packed = nd.scalarType == .int4
      }
      let blockwise = shape(f, "gate_s")?.count == 2
      let sShape: (Int, Int) -> [Int] = { r, w in blockwise ? [r, w / 32] : [r] }
      let want: [(String, [Int])] = [
        ("gate_q", [a, h]), ("gate_s", sShape(a, h)), ("up_q", [a, h]), ("up_s", sShape(a, h)),
        ("down_q", [h, a]), ("down_s", sShape(h, a)),
      ]
      for (input, expected) in want where shape(f, input) != expected {
        failure = "input \(input) is \(shape(f, input).map(String.init(describing:)) ?? "missing"), " +
                  "not \(expected)"
        sem.signal()
        return
      }
      let x = shape(f, "x")
      if x?.count != 2 || x?[1] != h {
        failure = "input x is \(x.map(String.init(describing:)) ?? "missing"), not [rows, \(h)]"
        sem.signal()
        return
      }
      guard let queue = device.makeCommandQueue() else {
        failure = "no command queue on the Metal device"
        sem.signal()
        return
      }
      let slot = Slot(model: model, function: f, queue: queue, event: event, hidden: h, ane: a,
                      packed: packed, blockwise: blockwise)
      // A cold function specializes on its first encode, which can outlast
      // the caller's job bound: burn one row through it here while load
      // latency is expected.
      do {
        try await warmUp(slot, on: device)
      } catch {
        failure = "the function did not warm up: \(error)"
        sem.signal()
        return
      }
      result = Unmanaged.passRetained(slot).toOpaque()
    } catch {
      failure = "\(error)"
    }
    sem.signal()
  }
  sem.wait()
  if let result { return result }
  return fail(failure ?? "an unknown error", error)
}

private func buffer(_ raw: UnsafeMutableRawPointer, _ offset: UInt64) -> (any MTLBuffer, Int) {
  (Unmanaged<AnyObject>.fromOpaque(raw).takeUnretainedValue() as! any MTLBuffer, Int(offset))
}

// CoreAI's stream command buffers hazard-track their bound buffers: a bound
// buffer whose Metal writes sit in the caller's still-open command buffer
// holds the evaluation for that buffer's end, deadlocking against the join
// that waits for the evaluation. An untracked alias over the same shared
// memory takes the buffer out of tracking; the job's wait/signal values on
// the shared event remain the ordering between the staging writes and the
// evaluation's reads.
private func untracked(_ raw: UnsafeMutableRawPointer, _ offset: UInt64, _ bytes: Int,
                       keep: inout [AnyObject]) -> (any MTLBuffer, Int) {
  let source = Unmanaged<AnyObject>.fromOpaque(raw).takeUnretainedValue() as! any MTLBuffer
  let contents = source.contents()
  guard let alias = source.device.makeBuffer(bytesNoCopy: contents.advanced(by: Int(offset)),
                                             length: bytes,
                                             options: [.storageModeShared, .hazardTrackingModeUntracked],
                                             deallocator: nil) else {
    FileHandle.standardError.write("coreai job: untracked alias failed, tracked fallback\n".data(using: .utf8)!)
    return (source, Int(offset))
  }
  keep.append(source)
  keep.append(alias)
  return (alias, 0)
}

@_cdecl("coreai_ffn_start")
public func coreai_ffn_start(_ ref: UnsafeMutableRawPointer?, _ job: UnsafeRawPointer?,
                             _ report: (@convention(c) (UnsafeMutableRawPointer?, Int32) -> Void)?,
                             _ ctx: UnsafeMutableRawPointer?) -> Int32 {
  guard let ref, let job, let report else { return 0 }
  let slot = Unmanaged<Slot>.fromOpaque(ref).takeUnretainedValue()
  let j = job.load(as: RawJob.self)
  let rows = Int(j.rows), h = slot.hidden, a = slot.ane

  // A wait-only command buffer completes without waiting for the event, so
  // the evaluation's release is the CPU-side notify the event itself
  // delivers once the GPU's signal lands (x packed, this layer's weights
  // staged). The job lock then keeps one evaluation on the shared x/y
  // buffers at a time, and run() only returns once its work has — the
  // event's `signal` goes up, the release the GPU joins on, and the job
  // reports.
  let listener = MTLSharedEventListener(dispatchQueue: DispatchQueue.global())
  slot.event.notify(listener, atValue: j.wait) { _, _ in
    Task {
      var ok = false
      slot.jobs.wait()
      do {
        var keep: [AnyObject] = [listener]
        var inputs: [String: InferenceFunction.AsyncValue] = [:]
        let (x, xo) = untracked(j.x.buffer, j.x.offset, rows * h * 2, keep: &keep)
        inputs["x"] = .init(unsafeBuffer: x, byteOffset: xo, scalarType: .float16, shape: [rows, h])
        let binds: [(String, coreai_ffn_buf, NDArray.ScalarType, [Int], Int)] = [
          ("gate_q", j.gate_q, slot.qType, [a, h], slot.qBytes(a, h)),
          ("gate_s", j.gate_s, .float16, slot.sShape(a, h), slot.sBytes(a, h)),
          ("up_q", j.up_q, slot.qType, [a, h], slot.qBytes(a, h)),
          ("up_s", j.up_s, .float16, slot.sShape(a, h), slot.sBytes(a, h)),
          ("down_q", j.down_q, slot.qType, [h, a], slot.qBytes(h, a)),
          ("down_s", j.down_s, .float16, slot.sShape(h, a), slot.sBytes(h, a)),
        ]
        for (name, b, type, dims, bytes) in binds {
          let (buf, off) = untracked(b.buffer, b.offset, bytes, keep: &keep)
          inputs[name] = .init(unsafeBuffer: buf, byteOffset: off, scalarType: type, shape: dims)
        }
        // encode() returns once the stream's work is submitted: the output
        // it hands back is the completion handle — awaiting the y value's
        // array is what makes the evaluation's end observable, after which
        // its rows are copied into the caller's y buffer.
        let result = try slot.function.encode(inputs: inputs, to: slot.stream)
        guard let y = result["y"], let out = try await y.ndArray else {
          throw NSError(domain: "coreai", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "no y output materialized"])
        }
        let (ybuf, yo) = untracked(j.y.buffer, j.y.offset, rows * h * 2, keep: &keep)
        let nd = out
        _ = nd.rawView().withUnsafeBytes { source, _, _ in
          memcpy(ybuf.contents().advanced(by: yo), source, rows * h * 2)
        }
        ok = true
      } catch {
        FileHandle.standardError.write("coreai job failed: \(error)\n".data(using: .utf8)!)
      }
      slot.event.signaledValue = j.signal
      slot.jobs.signal()
      report(ctx, ok ? 1 : 0)
    }
  }
  return 1
}

@_cdecl("coreai_ffn_close")
public func coreai_ffn_close(_ ref: UnsafeMutableRawPointer?) {
  guard let ref else { return }
  Unmanaged<Slot>.fromOpaque(ref).release()
}

// The C side's view of coreai_ffn_job (ops/CoreAiBridge.h).
public struct RawBuf { var buffer: UnsafeMutableRawPointer; var offset: UInt64 }
public struct RawJob {
  var x, y: RawBuf
  var gate_q, gate_s, up_q, up_s, down_q, down_s: RawBuf
  var wait, signal: UInt64
  var rows: UInt32
}
public typealias coreai_ffn_ref = UnsafeMutableRawPointer
public typealias coreai_ffn_buf = RawBuf
