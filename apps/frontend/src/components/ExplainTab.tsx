"use client";

import { useEffect, useState } from "react";
import { postExplain, type ExplanationResult } from "@/lib/api";
import { colorizeHeatmap, gradientCss } from "@/lib/heatmap";
import styles from "./ExplainTab.module.css";

/**
 * Explainability tab: requests a Seg-Grad-CAM heatmap for the current image
 * and shows it as a colour layer stacked over the original.
 *
 * The heatmap is fetched once (the call is gradient-based and heavier than
 * inference) and colourised once into a transparent overlay. Showing/hiding it
 * and changing its strength are pure CSS on the overlay layer, so they are
 * instant -- no per-pixel recompute on every interaction (the old opacity
 * slider re-ran a full-image pixel pass on each step, which is what felt laggy).
 */
export default function ExplainTab({
  file,
  previewUrl,
  explanation: explanationProp,
  onExplanation,
}: {
  file: File | null;
  previewUrl: string | null;
  explanation?: ExplanationResult | null;
  onExplanation?: (value: ExplanationResult | null) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Controlled-or-uncontrolled: when the parent supplies `explanation` +
  // `onExplanation` (the inference page does, backed by the session context),
  // the fetched explanation persists across tab switches and page navigation.
  // Rendered standalone (e.g. in unit tests) it falls back to its own state.
  const [explanationLocal, setExplanationLocal] =
    useState<ExplanationResult | null>(null);
  const explanation =
    explanationProp !== undefined ? explanationProp : explanationLocal;
  const setExplanation = onExplanation ?? setExplanationLocal;
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.85);

  // Colourise once when a new explanation arrives. Opacity is deliberately NOT
  // a dependency: it is applied as CSS on the overlay layer below, so dragging
  // it never re-runs this pixel pass. A new upload resets the component via its
  // `key` in the parent, so there is no reset effect here.
  useEffect(() => {
    if (!explanation) {
      return;
    }
    let cancelled = false;
    colorizeHeatmap(explanation.heatmap_b64)
      .then((url) => {
        if (!cancelled) setOverlayUrl(url);
      })
      .catch(() => {
        if (!cancelled) setOverlayUrl(null);
      });
    return () => {
      cancelled = true;
    };
  }, [explanation]);

  async function run() {
    if (!file) return;
    setLoading(true);
    setError(null);
    const fd = new FormData();
    fd.append("image", file);
    const { data, error: err } = await postExplain(fd);
    setLoading(false);
    if (err || !data) {
      setError(err ?? "Explanation failed.");
      return;
    }
    setExplanation(data);
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.intro}>
        <p className={styles.introText}>
          Seg-Grad-CAM highlights the regions that most drove the model&apos;s
          root classification. Warmer (yellow) areas contributed more.
        </p>
        <button
          id="btn-explain"
          className="btn btn-primary"
          disabled={!file || loading}
          onClick={run}
        >
          {loading ? (
            <>
              <span className="spinner spinner-sm" /> Generating...
            </>
          ) : explanation ? (
            "Regenerate explanation"
          ) : (
            "Generate explanation"
          )}
        </button>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginTop: "var(--space-4)" }}>
          {error}
        </div>
      )}

      {explanation && (
        <div className={`${styles.result} fade-in`}>
          <div className={`card ${styles.imageCard}`}>
            <div className={styles.imageStack}>
              {/* Base layer: the original plate. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl ?? ""}
                alt="Original plate"
                className={styles.image}
              />
              {/* Overlay layer: colourised heatmap. Opacity is pure CSS, so
                  the toggle and slider are instant. */}
              {overlayUrl && (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={overlayUrl}
                  alt="Grad-CAM heatmap overlay"
                  className={styles.overlayImg}
                  style={{ opacity: showOverlay ? opacity : 0 }}
                />
              )}
            </div>

            <div className={styles.controls}>
              <button
                type="button"
                className="btn btn-secondary"
                aria-pressed={showOverlay}
                onClick={() => setShowOverlay((v) => !v)}
              >
                {showOverlay ? "Hide overlay" : "Show overlay"}
              </button>

              <label htmlFor="range-alpha" className={styles.controlLabel}>
                Overlay strength
              </label>
              <input
                id="range-alpha"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                disabled={!showOverlay}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              <div className={styles.legend}>
                <span>low</span>
                <span
                  className={styles.legendBar}
                  style={{ background: gradientCss() }}
                />
                <span>high</span>
              </div>
            </div>
          </div>

          <div className={styles.meta}>
            <MetaItem label="Method" value={explanation.method} />
            <MetaItem label="Target layer" value={explanation.target_layer} />
            <MetaItem
              label="Peak attribution"
              value={explanation.heatmap_peak.toFixed(4)}
            />
            <MetaItem
              label="Processed"
              value={explanation.downscaled ? "downscaled" : "full resolution"}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.metaItem}>
      <span className={styles.metaLabel}>{label}</span>
      <span className={styles.metaValue}>{value}</span>
    </div>
  );
}
