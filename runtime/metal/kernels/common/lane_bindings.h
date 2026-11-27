#pragma once

#include "metal/abi/ExecutionGeometry.h"

// Batched kernels bind a per-lane tensor once per physical lane
// (ops::laneBindings). The production ABI has four lanes; the experimental
// width-8 ABI has eight. These macros generate the matching argument lists
// and select expressions at shader compile time, keeping the binding chain
// branch-free instead of turning it into a runtime array lookup.
#define RICHENGINE_LANE_CAT_INNER(a, b) a##b
#define RICHENGINE_LANE_CAT(a, b) RICHENGINE_LANE_CAT_INNER(a, b)

#if RICHENGINE_MAXIMUM_BATCH_WIDTH == 4
#define RICHENGINE_LANE_VALUES(name)                                          \
  RICHENGINE_LANE_CAT(name, 0), RICHENGINE_LANE_CAT(name, 1),                 \
      RICHENGINE_LANE_CAT(name, 2), RICHENGINE_LANE_CAT(name, 3)
#define RICHENGINE_LANE_BINDING(lane, lane0, lane1, lane2, lane3)             \
  ((lane) == 0 ? (lane0)                                                      \
               : ((lane) == 1 ? (lane1) : ((lane) == 2 ? (lane2) : (lane3))))
#define RICHENGINE_LANE_PARAMS(type, name)                                    \
  type RICHENGINE_LANE_CAT(name, 0), type RICHENGINE_LANE_CAT(name, 1),       \
      type RICHENGINE_LANE_CAT(name, 2), type RICHENGINE_LANE_CAT(name, 3)
#define RICHENGINE_LANE_BUFFER_ARGS(type, name, index)                        \
  type RICHENGINE_LANE_CAT(name, 0) [[buffer((index))]],                      \
      type RICHENGINE_LANE_CAT(name, 1) [[buffer((index) + 1)]],              \
      type RICHENGINE_LANE_CAT(name, 2) [[buffer((index) + 2)]],              \
      type RICHENGINE_LANE_CAT(name, 3) [[buffer((index) + 3)]]
#elif RICHENGINE_MAXIMUM_BATCH_WIDTH == 8
#define RICHENGINE_LANE_VALUES(name)                                          \
  RICHENGINE_LANE_CAT(name, 0), RICHENGINE_LANE_CAT(name, 1),                 \
      RICHENGINE_LANE_CAT(name, 2), RICHENGINE_LANE_CAT(name, 3),             \
      RICHENGINE_LANE_CAT(name, 4), RICHENGINE_LANE_CAT(name, 5),             \
      RICHENGINE_LANE_CAT(name, 6), RICHENGINE_LANE_CAT(name, 7)
#define RICHENGINE_LANE_BINDING(lane, lane0, lane1, lane2, lane3)             \
  ((lane) == 0 ? (lane0)                                                      \
               : ((lane) == 1 ? (lane1) : ((lane) == 2 ? (lane2) : (lane3))))
#define RICHENGINE_LANE_PARAMS(type, name)                                    \
  type RICHENGINE_LANE_CAT(name, 0), type RICHENGINE_LANE_CAT(name, 1),       \
      type RICHENGINE_LANE_CAT(name, 2), type RICHENGINE_LANE_CAT(name, 3),   \
      type RICHENGINE_LANE_CAT(name, 4), type RICHENGINE_LANE_CAT(name, 5),   \
      type RICHENGINE_LANE_CAT(name, 6), type RICHENGINE_LANE_CAT(name, 7)
#define RICHENGINE_LANE_BUFFER_ARGS(type, name, index)                        \
  type RICHENGINE_LANE_CAT(name, 0) [[buffer((index))]],                      \
      type RICHENGINE_LANE_CAT(name, 1) [[buffer((index) + 1)]],              \
      type RICHENGINE_LANE_CAT(name, 2) [[buffer((index) + 2)]],              \
      type RICHENGINE_LANE_CAT(name, 3) [[buffer((index) + 3)]],              \
      type RICHENGINE_LANE_CAT(name, 4) [[buffer((index) + 4)]],              \
      type RICHENGINE_LANE_CAT(name, 5) [[buffer((index) + 5)]],              \
      type RICHENGINE_LANE_CAT(name, 6) [[buffer((index) + 6)]],              \
      type RICHENGINE_LANE_CAT(name, 7) [[buffer((index) + 7)]]
#else
#error "RICHENGINE_MAXIMUM_BATCH_WIDTH supports production width 4 and experimental width 8"
#endif

#if RICHENGINE_MAXIMUM_BATCH_WIDTH == 4
#define RICHENGINE_LANE_SELECT_INNER(lane, name)                              \
  ((lane) == 0 ? RICHENGINE_LANE_CAT(name, 0)                                 \
               : ((lane) == 1 ? RICHENGINE_LANE_CAT(name, 1)                  \
                              : ((lane) == 2 ? RICHENGINE_LANE_CAT(name, 2)   \
                                             : RICHENGINE_LANE_CAT(name, 3))))
#else
#define RICHENGINE_LANE_SELECT_INNER(lane, name)                              \
  ((lane) == 0                                                                \
       ? RICHENGINE_LANE_CAT(name, 0)                                         \
       : ((lane) == 1                                                         \
              ? RICHENGINE_LANE_CAT(name, 1)                                  \
              : ((lane) == 2                                                  \
                     ? RICHENGINE_LANE_CAT(name, 2)                           \
                     : ((lane) == 3                                           \
                            ? RICHENGINE_LANE_CAT(name, 3)                    \
                            : ((lane) == 4                                    \
                                   ? RICHENGINE_LANE_CAT(name, 4)             \
                                   : ((lane) == 5                             \
                                          ? RICHENGINE_LANE_CAT(name, 5)      \
                                          : ((lane) == 6                      \
                                                 ? RICHENGINE_LANE_CAT(name, 6) \
                                                 : RICHENGINE_LANE_CAT(name, 7))))))))
#endif
#define RICHENGINE_LANE_SELECT(lane, name)                                    \
  RICHENGINE_LANE_SELECT_INNER(lane, name)
