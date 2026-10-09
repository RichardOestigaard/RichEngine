import { Show } from "solid-js";
import type { JSX } from "solid-js";
import {
  AlertDialog,
  AlertDialogDescription,
  AlertDialogOverlay,
  AlertDialogPanel,
  AlertDialogTitle,
} from "terracotta";

/* Shared destructive-action confirmation built on terracotta's AlertDialog:
   role="alertdialog", focus trap, Esc/overlay-click close, and focus restore
   to the triggering button. Styles (.confirm-*) live in styles.css. */
interface ConfirmProps {
  open: boolean;
  title: string;
  children?: JSX.Element;
  confirmLabel?: string;
  confirmClass?: string;
  busy?: boolean;
  onConfirm: () => void;
  /* Optional middle action — for a two-choice dialog like Tune's
     Quick/Complete mode pick. */
  secondaryLabel?: string;
  secondaryClass?: string;
  onSecondary?: () => void;
  onClose: () => void;
}

export default function Confirm(props: ConfirmProps) {
  return (
    <AlertDialog
      isOpen={props.open}
      onChange={(open) => {
        if (!open) props.onClose();
      }}
    >
      <AlertDialogOverlay class="confirm-overlay" />
      <AlertDialogPanel class="confirm-panel">
        <AlertDialogTitle as="div" class="confirm-title">
          {props.title}
        </AlertDialogTitle>
        <AlertDialogDescription class="confirm-desc">
          {props.children}
        </AlertDialogDescription>
        <div class="confirm-actions">
          <button type="button" class="btn" onClick={props.onClose}>
            Cancel
          </button>
          <Show when={props.secondaryLabel}>
            <button
              type="button"
              class={props.secondaryClass ?? "btn"}
              disabled={props.busy}
              onClick={props.onSecondary}
            >
              {props.secondaryLabel}
            </button>
          </Show>
          <button
            type="button"
            class={props.confirmClass ?? "btn danger"}
            disabled={props.busy}
            onClick={props.onConfirm}
          >
            {props.confirmLabel ?? "Confirm"}
          </button>
        </div>
      </AlertDialogPanel>
    </AlertDialog>
  );
}
