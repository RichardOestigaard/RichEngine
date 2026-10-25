import { createSignal } from "solid-js";

/* True while the served model is a d1-family decision model: generation
   pages are unavailable and /v1/systemone + /v1/judgments run the native
   decision pathway. Reported per model entry as `decision` by /v1/models. */
const [decisionModel, setDecisionModel] = createSignal(false);

export { decisionModel, setDecisionModel };
