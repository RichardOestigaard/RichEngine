#pragma once

#include "engine/wire/Types.hpp"
#include "model/Model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// The engine decodes client frames and encodes engine events;
// server/protocol.py does the reverse. dev/tests/engine/protocol_golden.txt
// pins the bytes both sides agree on.
namespace richengine::protocol {

inline constexpr uint16_t kProtocolVersion = 7;
inline constexpr size_t kFrameHeaderBytes = 24;
inline constexpr uint32_t kStatusSchemaVersion = 7;
// Image pixels travel inside the request frame; a multi-image agent turn can
// carry well over 64 MiB of resized RGB bytes.
inline constexpr uint64_t kAbsoluteMaxFramePayloadBytes =
    256ULL * 1024 * 1024;

// Every integer, including binary prompt/token words, is little-endian on the
// wire.  Header flags and reserved bytes must be zero in native protocol.
enum class FrameType : uint16_t {
  Request = 0x0001,
  Cancel = 0x0002,
  MaskResponse = 0x0003,
  StatusRequest = 0x0004,
  ForcedTokens = 0x0005,
  ConstraintDfa = 0x0006,
  Vocabulary = 0x0007,
  ScoreBatch = 0x0008,
  CorpusAppend = 0x0009,

  Ready = 0x0100,
  Start = 0x0101,
  Tokens = 0x0102,
  MaskRequest = 0x0103,
  Done = 0x0104,
  Error = 0x0105,
  StatusJson = 0x0107,
  PromptProgress = 0x0108,
};

// Request errors reject one request while preserving the stream.  An
// engine-unhealthy error asks the supervisor to replace the engine.  A
// protocol-fatal error means framing is no longer trusted, so the stream must
// close without attempting another framing mode.
enum class FailureClass : uint8_t {
  RequestError = 1,
  EngineUnhealthy = 2,
  ProtocolFatal = 3,
};

[[nodiscard]] bool connectionMustClose(FailureClass failureClass);

enum class IssueCode : uint16_t {
  None = 0,
  BadMagic,
  UnsupportedVersion,
  InvalidHeaderSize,
  UnknownFrameType,
  NonZeroHeaderFlags,
  NonZeroReservedField,
  FrameTooLarge,
  InvalidPayloadLength,
  TruncatedFrame,
  InvalidRequestId,
  InvalidEnumValue,
  InvalidDeadline,
  InvalidSampling,
  InvalidCount,
  InvalidConstraint,
  InvalidErrorClassification,
  LimitExceeded,
  IntegerOverflow,
  AllocationFailure,
};

[[nodiscard]] std::string_view issueCodeName(IssueCode code);

struct ProtocolIssue {
  FailureClass failureClass = FailureClass::ProtocolFatal;
  IssueCode code = IssueCode::None;
  uint64_t requestId = 0;
  std::string message;

  [[nodiscard]] std::string describe() const;
  bool operator==(const ProtocolIssue &) const = default;
};

template <typename T> struct ProtocolResult {
  std::optional<T> value;
  std::optional<ProtocolIssue> issue;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && !issue.has_value();
  }
};

// Valid as validateLimits checks them: NativeRuntime checks its limits once,
// and the codec and the frame parser rely on that.
struct ProtocolLimits {
  uint64_t maxFramePayloadBytes = kAbsoluteMaxFramePayloadBytes;
  uint64_t maxStatusJsonBytes = 32ULL * 1024 * 1024;
  uint64_t maxErrorStringBytes = 1ULL * 1024 * 1024;
  uint32_t maxPromptTokens = 1U << 20;
  uint32_t maxLogicalOutputTokens = 1U << 20;
  uint32_t maxTokenBatch = 4096;
  uint32_t maxSimulationTokens = 32;
  uint32_t maxMaskWords = 1U << 20;
  uint32_t maxImageSpans = 64;
};

[[nodiscard]] std::optional<ProtocolIssue>
validateLimits(const ProtocolLimits &limits);

// A request payload starts with these fields, at these byte offsets:
//    0 u64 requestId
//    8 u8  priority
//    9 u8  constraint
//   10 u64 absoluteDeadlineUnixMicros
//   18 u64 remainingDeadlineMicros
//   26 u32 logicalMaxOutputTokens
//   30 u32 prompt token count
//   34 u32 image span count
//   38 sampling: f32 temperature, f32 topP, u32 topK, f32 presencePenalty,
//      f32 frequencyPenalty, f32 repetitionPenalty, f32 minP, u64 seed
//   74 u8  returnProgress
//   75 u32 score token count
//   79 u32 generationPromptTokens
//   83 u32 flags
//   87 u32 sharedPrefixTokens
// then the prompt tokens, the 32-byte image spans, the image pixels and the
// score tokens.
inline constexpr uint64_t kRequestFixedBytes = 91;

struct RequestFrame {
  uint64_t requestId = 0;
  engine::RequestPriority priority = engine::RequestPriority::Normal;

  // absoluteDeadlineUnixMicros is wall-clock UTC. remainingDeadlineMicros
  // is the sender's remaining budget at serialization time.  Admission
  // should honor the earlier of the two after accounting for transit time.
  uint64_t absoluteDeadlineUnixMicros = 0;
  uint64_t remainingDeadlineMicros = 0;

  uint32_t logicalMaxOutputTokens = 0;
  std::vector<uint32_t> promptTokens;
  // Sorted, non-overlapping image spans and their resized uint8 RGB pixels,
  // concatenated in span order (gridHeight*16 x gridWidth*16 x 3 each).
  // Both are empty for text-only requests.
  std::vector<ImageSpan> imageSpans;
  std::vector<uint8_t> imagePixels;
  // The defaults are greedy selection with nothing changing the logits,
  // which score requests require (seed aside).
  SamplingParameters sampling;
  ConstraintMode constraint = ConstraintMode::None;
  bool returnProgress = false;
  // Empty selects ordinary generation. Nonempty selects score-only mode:
  // 2..255 distinct token ids, logicalMaxOutputTokens must be zero, and the
  // request must be text-only, unconstrained, and greedy.
  std::vector<uint32_t> scoreTokens{};
  // Trailing prompt tokens of the chat template's generation prompt; zero
  // when unknown. It must leave at least one prompt token.
  uint32_t generationPromptTokens = 0;
  // RequestFlag bits.
  uint32_t flags = 0;
  // Leading prompt tokens that later requests are expected to share, such as
  // the chat template's system prompt and tools; zero when unknown. It must
  // not exceed the prompt.
  uint32_t sharedPrefixTokens = 0;

  bool operator==(const RequestFrame &) const = default;
};

// The other payloads, in wire order; counts precede what they count, and
// the variable parts come last:
//   Cancel         u64 requestId
//   MaskResponse   u64 requestId, u64 maskRequestId, u32 count, u32 words
//   StatusRequest  u64 correlationId
//   Ready          u32 maxConcurrentRequests, u32 maxContextTokens, u8 vision
//   Start          u64 requestId, u32 lane, u32 matchedPromptTokens
//   PromptProgress u64 requestId, u32 processedTokens, u64 elapsedMicros
//   Tokens         u64 requestId, u32 sequenceOffset, u32 count, u32 tokens
//   MaskRequest    u64 requestId, u64 maskRequestId, u32 wordsPerMask,
//                  u32 count, u32 simulation tokens
//   ConstraintDfa  u64 requestId, u32 nStates, u32 flags,
//                  i16 table[nStates*256], u8 accepting[ceil(nStates/8)],
//                  u32 eosCount, u32 eosIds[eosCount],
//                  u32 markerBytes, u8 marker[nStates*32]
//   Vocabulary     u32 vocab, u32 token byte count, u8 token bytes,
//                  i32 offsets[vocab + 1],
//                  u32 specBitsBytes, u8 specBits[ceil(vocab/8)]
//   Done           u64 requestId, u8 reason, u32 promptTokens,
//                  u32 completionTokens, u64 prefillMicros, u64 decodeMicros,
//                  u64 wallMicros, u32 draftedTokens, u32 acceptedDraftTokens,
//                  u32 count, f32 option logits
//   Error          u8 failureClass, u8 retryable, u64 requestId, u32 code
//                  bytes, u32 message bytes, the code, the message
//   StatusJson     u64 correlationId, the JSON
struct CancelFrame {
  uint64_t requestId = 0;

  bool operator==(const CancelFrame &) const = default;
};

struct MaskResponseFrame {
  uint64_t requestId = 0;
  uint64_t maskRequestId = 0;
  std::vector<uint32_t> maskWords;

  bool operator==(const MaskResponseFrame &) const = default;
};

struct StatusRequestFrame {
  uint64_t correlationId = 0;

  bool operator==(const StatusRequestFrame &) const = default;
};

// Grammar-forced tokens the constraint computed at the last committed
// position: tokens[0] sits where the lane's pending anchor does, so the span
// applies only while they are equal. The rest verify under their masks.
struct ForcedTokensFrame {
  uint64_t requestId = 0;
  std::vector<uint32_t> tokens;

  bool operator==(const ForcedTokensFrame &) const = default;
};

// A standalone token stream pushed into the cross-request n-gram corpus
// (RICHENGINE_NGRAM_CORPUS): workspace files, tool outputs, or
// emission-format copies indexed ahead of the request that re-emits
// them. Process-wide, request-agnostic.
struct CorpusAppendFrame {
  // Session tag the corpus doc inherits: 0 = workspace/global, nonzero =
  // the caller's conversation key (opencode sessionID, a prompt hash, …).
  uint64_t session = 0;
  std::vector<uint32_t> tokens;

  bool operator==(const CorpusAppendFrame &) const = default;
};

// A compiled byte-DFA for one request's grammar: while the engine holds it,
// decode computes the lane's masks on the GPU (kernels/decode/dfa.metal)
// and no MaskRequest event goes out. The table is row-major Q x 256 with
// -1 for a dead transition (the host prunes edges that cannot reach an
// accepting state, so a non-dead end always stays live). accepting is the
// EOS-admitting bitset — the grammar's accepting ∪ free-text states;
// eosIds is its grammar-EOS set (≤ 8 ids); marker is 32 bytes per state,
// bit b of state s marking edge (s, b) special-only. costs, when present,
// is one i16 per state — the minimum tokens a state's grammar owes before
// it can terminate (32767: no bounded walk); the decode budget rule gates
// a lane's mask rows with it, and empty costs keep the plain walk.
struct ConstraintDfaFrame {
  uint64_t requestId = 0;
  std::vector<int16_t> table;
  std::vector<uint8_t> accepting;
  std::vector<uint32_t> eosIds;
  std::vector<uint8_t> marker;
  std::vector<int16_t> costs;

  bool operator==(const ConstraintDfaFrame &) const = default;
};

// The packed token vocabulary the DFA walk kernels index: the bytes of
// every vocabulary token behind vocab + 1 prefix offsets; equal offsets
// mark a dead token the mask never admits; specBits bit t names token t
// special for the marker rule. Process-wide: applies to every request
// after it, and is resent after an engine restart.
struct VocabularyFrame {
  std::vector<uint8_t> tokenBytes;
  std::vector<int32_t> tokenOffsets;
  std::vector<uint8_t> specialBits;

  bool operator==(const VocabularyFrame &) const = default;
};

struct ReadyEvent {
  uint32_t maxConcurrentRequests = 0;
  uint32_t maxContextTokens = 0;
  // Requests may carry image spans. A model serving without vision rejects
  // each image request with a request error.
  bool vision = false;

  bool operator==(const ReadyEvent &) const = default;
};

struct StartEvent {
  uint64_t requestId = 0;
  uint32_t lane = 0;
  // The cached prefix the request starts from; zero for a cold start.
  uint32_t matchedPromptTokens = 0;

  bool operator==(const StartEvent &) const = default;
};

struct PromptProgressEvent {
  uint64_t requestId = 0;
  uint32_t processedTokens = 0;
  uint64_t elapsedMicros = 0;

  bool operator==(const PromptProgressEvent &) const = default;
};

struct TokensEvent {
  uint64_t requestId = 0;
  uint32_t sequenceOffset = 0;
  std::vector<uint32_t> tokens;

  bool operator==(const TokensEvent &) const = default;
};

struct MaskRequestEvent {
  uint64_t requestId = 0;
  uint64_t maskRequestId = 0;
  uint32_t wordsPerMask = 0;
  std::vector<uint32_t> simulationTokens;

  // Rows describe the constraint state before and after each simulated
  // token. Empty input is the initial-anchor request (one row); verification
  // passes [pending_anchor, draft...] and therefore requires len + 1 rows.
  bool operator==(const MaskRequestEvent &) const = default;
};

struct DoneEvent {
  uint64_t requestId = 0;
  engine::EngineFinishReason reason = engine::EngineFinishReason::Length;
  uint32_t promptTokens = 0;
  uint32_t completionTokens = 0;
  uint64_t prefillMicros = 0;
  uint64_t decodeMicros = 0;
  uint64_t wallMicros = 0;
  // The request's lifetime speculative counts: proposals the verify stages
  // read, and the share of them accepted into the output.
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  // Raw final-prompt-position logits at the request's scoreTokens, in
  // requested order. Empty for generation and for cancelled/failed scoring.
  std::vector<float> optionLogits{};

  bool operator==(const DoneEvent &) const = default;
};

struct ErrorEvent {
  FailureClass failureClass = FailureClass::RequestError;
  uint64_t requestId = 0;
  bool retryable = false;
  std::string code;
  std::string message;

  bool operator==(const ErrorEvent &) const = default;
};

// JSON is deliberately opaque to the transport: the document carries its
// own schema_version, and the frame length carries the exact JSON byte count
// (including whitespace) without line or C-string assumptions.
struct StatusJsonEvent {
  uint64_t correlationId = 0;
  std::string json;

  bool operator==(const StatusJsonEvent &) const = default;
};

// A batch of score-only requests sharing one prompt prefix — the decision
// pathway's questions over one state. Each item expands to a score request
// with prompt = prefixTokens + suffixTokens, sharedPrefixTokens =
// prefixTokens.size(), greedy unconstrained sampling and no output budget.
// Wire layout: u64 absolute deadline, u64 remaining deadline, u8 priority,
// u8 flags, u32 item count, u32 prefix token count, u32 prefix tokens; then
// per item u64 requestId, u32 suffix count, u32 score count, u32 suffix
// tokens, u32 score tokens.
struct ScoreBatchFrame {
  struct Item {
    uint64_t requestId = 0;
    std::vector<uint32_t> suffixTokens;
    std::vector<uint32_t> scoreTokens;

    bool operator==(const Item &) const = default;
  };

  uint64_t absoluteDeadlineUnixMicros = 0;
  uint64_t remainingDeadlineMicros = 0;
  engine::RequestPriority priority = engine::RequestPriority::Normal;
  uint8_t flags = 0;
  std::vector<uint32_t> prefixTokens;
  std::vector<Item> items;

  bool operator==(const ScoreBatchFrame &) const = default;
};

using ClientMessage =
    std::variant<RequestFrame, CancelFrame, MaskResponseFrame,
                 StatusRequestFrame, ForcedTokensFrame, ConstraintDfaFrame,
                 VocabularyFrame, ScoreBatchFrame, CorpusAppendFrame>;
using EngineEvent =
    std::variant<ReadyEvent, StartEvent, PromptProgressEvent, TokensEvent,
                 MaskRequestEvent, DoneEvent, ErrorEvent, StatusJsonEvent>;

struct Frame {
  FrameType type = FrameType::Request;
  std::vector<uint8_t> payload;

  bool operator==(const Frame &) const = default;
};

// Takes the frame: a request's image pixels stay in its payload's buffer.
[[nodiscard]] ProtocolResult<ClientMessage>
decodeFrame(Frame &&frame, const ProtocolLimits &limits);
// The whole frame, header included. An event that breaks its rules is the
// engine's defect: the issue is EngineUnhealthy.
[[nodiscard]] ProtocolResult<std::vector<uint8_t>>
serializeEvent(const EngineEvent &event, const ProtocolLimits &limits);

struct ParseStep {
  size_t consumedBytes = 0;
  std::optional<Frame> frame;
  std::optional<ProtocolIssue> issue;
};

// Incremental one-frame-at-a-time parser of client frames.  consume() stops
// as soon as it yields one complete frame, so callers can process arbitrarily
// long streams without retaining a batch of frames.  An event type is an
// unknown frame type.  finish() must be called at EOF to turn a partial
// header or payload into a protocol-fatal truncation.
class FrameParser {
public:
  explicit FrameParser(ProtocolLimits limits);

  [[nodiscard]] ParseStep consume(std::span<const uint8_t> bytes);
  [[nodiscard]] std::optional<ProtocolIssue> finish();

private:
  [[nodiscard]] std::optional<ProtocolIssue> parseHeader();
  [[nodiscard]] ParseStep fail(size_t consumed, ProtocolIssue issue);
  void resetCurrentFrame();

  ProtocolLimits limits_;
  std::array<uint8_t, kFrameHeaderBytes> header_{};
  size_t headerBytes_ = 0;
  bool readingPayload_ = false;
  FrameType currentType_ = FrameType::Request;
  uint64_t expectedPayloadBytes_ = 0;
  std::vector<uint8_t> payload_;
  std::optional<ProtocolIssue> terminalIssue_;
};

} // namespace richengine::protocol
