"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { InferenceResult, ExplanationResult } from "@/lib/api";

/** The three tabs of the inference result view. */
export type InferenceTab = "result" | "explain" | "metrics";

/**
 * The full inference "session": everything the user produces on the inference
 * page that should outlive a tab switch or a trip to the dashboard.
 *
 * It lives in a provider mounted in the root layout - above the router - so
 * navigating between pages (which unmounts the page component, and with it any
 * page-local useState) does not discard the result or the explanation. This is
 * the difference between page state and app state: anything here is app state.
 */
export interface InferenceSession {
  file: File | null;
  previewUrl: string | null;
  plateId: string;
  result: InferenceResult | null;
  overlayUrl: string | null;
  activeTab: InferenceTab;
  feedbackFlag: string;
  feedbackNotes: string;
  feedbackSubmitted: boolean;
  explanation: ExplanationResult | null;
}

const INITIAL: InferenceSession = {
  file: null,
  previewUrl: null,
  plateId: "",
  result: null,
  overlayUrl: null,
  activeTab: "result",
  feedbackFlag: "good",
  feedbackNotes: "",
  feedbackSubmitted: false,
  explanation: null,
};

interface InferenceContextValue {
  session: InferenceSession;
  /** Shallow-merge a partial update into the session. */
  patch: (partial: Partial<InferenceSession>) => void;
  /** Reset the whole session back to its initial empty state. */
  reset: () => void;
}

const InferenceContext = createContext<InferenceContextValue | null>(null);

/**
 * Provides the persistent inference session to the whole app.
 *
 * A single object held in one useState (rather than a dozen separate states)
 * keeps the API tiny - `patch` for any field, `reset` for a clean slate - and
 * easy to extend later without touching every consumer.
 *
 * @param props - Component props.
 * @param props.children - The app subtree that can read the session.
 * @returns The provider element.
 */
export function InferenceProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<InferenceSession>(INITIAL);

  const patch = useCallback((partial: Partial<InferenceSession>) => {
    setSession((prev) => ({ ...prev, ...partial }));
  }, []);

  const reset = useCallback(() => setSession(INITIAL), []);

  // Memoise so consumers only re-render when the session actually changes,
  // not on every render of the provider's parent.
  const value = useMemo(
    () => ({ session, patch, reset }),
    [session, patch, reset],
  );

  return (
    <InferenceContext.Provider value={value}>
      {children}
    </InferenceContext.Provider>
  );
}

/**
 * Access the persistent inference session. Must be called inside an
 * InferenceProvider (mounted in the root layout).
 *
 * @returns The session and its `patch` / `reset` helpers.
 */
export function useInferenceSession(): InferenceContextValue {
  const ctx = useContext(InferenceContext);
  if (!ctx) {
    throw new Error(
      "useInferenceSession must be used within an InferenceProvider",
    );
  }
  return ctx;
}
