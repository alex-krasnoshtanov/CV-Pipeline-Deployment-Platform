"use client";

import {
  ChangeEvent,
  DragEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { useInferenceSession } from "@/context/InferenceContext";
import {
  getHealth,
  postInfer,
  postFeedback,
  type HealthData,
  type ExplanationResult,
} from "@/lib/api";
import { overlayMask } from "@/lib/overlay";
import LandmarkOverlay from "@/components/LandmarkOverlay";
import ExplainTab from "@/components/ExplainTab";
import MetricsTab from "@/components/MetricsTab";
import Header from "@/components/Header";
import styles from "./page.module.css";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/tiff"];


/**
 * Main dashboard page. Protected — redirects to `/login` if not authenticated.
 *
 * Provides image upload, inference execution, and a tabbed result view:
 * Result (mask overlay + root-tip circles + landmark table + feedback),
 * Explain (Seg-Grad-CAM heatmap), and Metrics (confidence vs alert thresholds).
 */
export default function DashboardPage() {
  const router = useRouter();
  const { user, loading: authLoading, logout } = useAuth();

  /* ---------- health ---------- */
  const [health, setHealth] = useState<HealthData | null>(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      const { data } = await getHealth();
      if (!cancelled) {
        setHealth(data);
        setHealthError(!data);
      }
    }
    poll();
    const id = setInterval(poll, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  /* ---------- persistent inference session ----------
     Lives in a layout-level provider, so it survives switching tabs AND
     navigating to the dashboard and back (which unmounts this page). */
  const { session, patch } = useInferenceSession();
  const {
    file,
    previewUrl,
    plateId,
    result,
    overlayUrl,
    activeTab,
    feedbackFlag,
    feedbackNotes,
    feedbackSubmitted,
  } = session;

  // Stable setter for the Explain tab; persists the fetched explanation into
  // the session so it is not lost on a tab switch or navigation.
  const setExplanation = useCallback(
    (value: ExplanationResult | null) => patch({ explanation: value }),
    [patch],
  );

  /* ---------- transient UI state (fine to reset on navigation) ---------- */
  const [inferring, setInferring] = useState(false);
  const [inferError, setInferError] = useState<string | null>(null);
  const [showTips, setShowTips] = useState(true);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (f: File | null) => {
      // New (or cleared) upload: drop every derived artefact so a stale result,
      // overlay or explanation never shows against a different image.
      patch({
        file: f,
        previewUrl: f ? URL.createObjectURL(f) : null,
        result: null,
        overlayUrl: null,
        explanation: null,
        feedbackSubmitted: false,
        activeTab: "result",
      });
    },
    [patch],
  );

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f && ACCEPTED_TYPES.includes(f.type)) {
      handleFile(f);
    }
  }

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    handleFile(e.target.files?.[0] ?? null);
  }

  async function runInference() {
    if (!file) return;
    setInferring(true);
    setInferError(null);
    patch({
      result: null,
      overlayUrl: null,
      feedbackSubmitted: false,
      activeTab: "result",
    });

    const fd = new FormData();
    fd.append("image", file);
    if (plateId.trim()) fd.append("plate_id", plateId.trim());

    const { data, error, status } = await postInfer(fd);

    if (status === 401) {
      // Session expired.
      await logout();
      router.replace("/login");
      return;
    }

    if (error || !data) {
      setInferError(error ?? "Inference failed.");
      setInferring(false);
      return;
    }

    patch({ result: data });
    setInferring(false);

    // Build overlay asynchronously after result is set.
    if (previewUrl && data.mask_b64) {
      try {
        const url = await overlayMask(previewUrl, data.mask_b64);
        patch({ overlayUrl: url });
      } catch {
        // Overlay failed — still show results without it.
      }
    }
  }

  async function submitFeedback() {
    if (!result?.prediction_id) return;
    setFeedbackSubmitting(true);
    setFeedbackError(null);

    const { error } = await postFeedback({
      prediction_id: result.prediction_id,
      flag: feedbackFlag,
      notes: feedbackNotes.trim() || undefined,
    });

    setFeedbackSubmitting(false);
    if (error) {
      setFeedbackError(error);
    } else {
      patch({ feedbackSubmitted: true });
    }
  }

  /* ---------- auth gate ---------- */
  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  if (authLoading || !user) {
    return (
      <div className={styles.loadingOverlay}>
        <span className="spinner" />
      </div>
    );
  }

  /* ---------- health indicator ---------- */
  const healthDotClass = healthError
    ? styles.healthDotError
    : health?.model_loaded
      ? styles.healthDotOk
      : styles.healthDotLoading;

  const healthLabel = healthError
    ? "Backend unreachable"
    : health?.model_loaded
      ? `Model ${health.model_version} | Pipeline ${health.pipeline_version} | ${health.serving_mode}`
      : "Model loading...";

  return (
    <div className={styles.page}>
      <Header active="inference" />

      {/* ---- main ---- */}
      <main className={styles.main}>
        {/* health bar */}
        <div className={styles.healthBar}>
          <span className={`${styles.healthDot} ${healthDotClass}`} />
          <span>{healthLabel}</span>
        </div>

        {/* upload section */}
        <section className={styles.uploadSection}>
          <h2 className={styles.sectionTitle}>Inference</h2>

          <div className={styles.uploadRow}>
            <div
              className={`${styles.dropZone} ${dragOver ? styles.dropZoneDragOver : ""} ${file ? styles.dropZoneActive : ""}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <input
                ref={fileRef}
                id="input-image"
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff"
                style={{ display: "none" }}
                onChange={onFileChange}
              />
              <div className={styles.dropLabel}>
                {file ? "Click to change image" : "Drop a plate image here or click to browse"}
              </div>
              {!file && <div className={styles.dropHint}>PNG, JPG, or TIFF</div>}
              {file && <div className={styles.fileName}>{file.name}</div>}
              {previewUrl && !result && (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={previewUrl}
                  alt="Upload preview"
                  className={styles.previewThumb}
                />
              )}
            </div>

            <div className={styles.plateField}>
              <label htmlFor="input-plate-id">Plate ID (optional)</label>
              <input
                id="input-plate-id"
                type="text"
                placeholder="e.g. PLATE-001"
                value={plateId}
                onChange={(e) => patch({ plateId: e.target.value })}
              />
            </div>
          </div>

          <div className={styles.actionBar}>
            <button
              id="btn-infer"
              className="btn btn-primary"
              disabled={!file || inferring || !health?.model_loaded}
              onClick={runInference}
            >
              {inferring ? (
                <>
                  <span className="spinner spinner-sm" /> Running...
                </>
              ) : (
                "Run Inference"
              )}
            </button>
            {file && (
              <button className="btn btn-ghost" onClick={() => handleFile(null)}>
                Clear
              </button>
            )}
          </div>

          {inferError && (
            <div className="alert alert-error" style={{ marginTop: "var(--space-4)" }}>
              {inferError}
            </div>
          )}
        </section>

        {/* ---- results ---- */}
        {result && (
          <section className={`${styles.resultsSection} fade-in`}>
            <hr className="divider" />

            {/* tab bar */}
            <div className={styles.tabs} role="tablist">
              <button
                id="tab-result"
                role="tab"
                aria-selected={activeTab === "result"}
                className={`${styles.tab} ${activeTab === "result" ? styles.tabActive : ""}`}
                onClick={() => patch({ activeTab: "result" })}
              >
                Result
              </button>
              <button
                id="tab-explain"
                role="tab"
                aria-selected={activeTab === "explain"}
                className={`${styles.tab} ${activeTab === "explain" ? styles.tabActive : ""}`}
                onClick={() => patch({ activeTab: "explain" })}
              >
                Explain
              </button>
              <button
                id="tab-metrics"
                role="tab"
                aria-selected={activeTab === "metrics"}
                className={`${styles.tab} ${activeTab === "metrics" ? styles.tabActive : ""}`}
                onClick={() => patch({ activeTab: "metrics" })}
              >
                Metrics
              </button>
            </div>

            {/* ---- Result tab ---- */}
            {activeTab === "result" && (
              <div className="fade-in">
                <div className={styles.imagesRow}>
                  <div className={`card ${styles.imageCard}`}>
                    <div className={styles.imageCardTitle}>Mask</div>
                    {result.mask_b64 ? (
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img
                        src={`data:image/png;base64,${result.mask_b64}`}
                        alt="Segmentation mask"
                        className={styles.imageCardImg}
                      />
                    ) : (
                      <div className="skeleton" style={{ width: "100%", height: 200 }} />
                    )}
                  </div>
                  <div className={`card ${styles.imageCard}`}>
                    <div className={styles.imageCardHead}>
                      <div className={styles.imageCardTitle}>Combined</div>
                      {result.landmarks.length > 0 && (
                        <label className={styles.tipToggle}>
                          <input
                            type="checkbox"
                            checked={showTips}
                            onChange={(e) => setShowTips(e.target.checked)}
                          />
                          Root tips
                        </label>
                      )}
                    </div>
                    {overlayUrl ? (
                      <div className={styles.overlayStage}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={overlayUrl}
                          alt="Combined overlay"
                          className={styles.imageCardImg}
                        />
                        {showTips && result.landmarks.length > 0 && (
                          <LandmarkOverlay
                            landmarks={result.landmarks}
                            width={result.image_width_px}
                            height={result.image_height_px}
                          />
                        )}
                      </div>
                    ) : (
                      <div className="skeleton" style={{ width: "100%", height: 200 }} />
                    )}
                  </div>
                </div>

                {/* landmarks table */}
                {result.landmarks.length > 0 && (
                  <div className={styles.landmarksSection}>
                    <h3 className={styles.sectionTitle}>Landmarks</h3>
                    <div className={`card table-container`}>
                      <table>
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>X</th>
                            <th>Y</th>
                            <th>Confidence</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.landmarks.map((lm) => (
                            <tr key={lm.id}>
                              <td>{lm.id}</td>
                              <td>{lm.x}</td>
                              <td>{lm.y}</td>
                              <td>{lm.confidence.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* feedback */}
                {result.prediction_id == null ? (
                  <p
                    className={styles.userMeta}
                    style={{ marginTop: "var(--space-4)" }}
                  >
                    Feedback unavailable — prediction was not saved to database.
                  </p>
                ) : (
                  <div className={styles.feedbackSection}>
                    <h3 className={styles.sectionTitle}>Feedback</h3>

                    {feedbackSubmitted ? (
                      <div className="alert alert-success">
                        Feedback submitted successfully.
                      </div>
                    ) : (
                      <>
                        <div className={styles.feedbackRow}>
                          <div className={styles.feedbackSelect}>
                            <label htmlFor="select-flag">Assessment</label>
                            <select
                              id="select-flag"
                              value={feedbackFlag}
                              onChange={(e) => patch({ feedbackFlag: e.target.value })}
                            >
                              <option value="good">Good</option>
                              <option value="bad">Bad</option>
                              <option value="uncertain">Uncertain</option>
                            </select>
                          </div>
                          <div className={styles.feedbackNotes}>
                            <label htmlFor="textarea-notes">Notes (optional)</label>
                            <textarea
                              id="textarea-notes"
                              placeholder="e.g. Root tip 2 is misplaced, model missed the lateral root."
                              maxLength={2000}
                              value={feedbackNotes}
                              onChange={(e) => patch({ feedbackNotes: e.target.value })}
                            />
                          </div>
                        </div>

                        {feedbackError && (
                          <div
                            className="alert alert-error"
                            style={{ marginTop: "var(--space-3)" }}
                          >
                            {feedbackError}
                          </div>
                        )}

                        <div className={styles.feedbackActions}>
                          <button
                            id="btn-feedback"
                            className="btn btn-secondary"
                            disabled={feedbackSubmitting}
                            onClick={submitFeedback}
                          >
                            {feedbackSubmitting ? (
                              <>
                                <span className="spinner spinner-sm" /> Submitting...
                              </>
                            ) : (
                              "Submit Feedback"
                            )}
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ---- Explain tab ----
                 Kept mounted (just hidden) so its colourised heatmap and view
                 controls survive tab switches without recomputing; the fetched
                 explanation lives in the session, so it also survives navigating
                 away to the dashboard and back. */}
            <div style={{ display: activeTab === "explain" ? "block" : "none" }}>
              <ExplainTab
                key={previewUrl ?? "no-file"}
                file={file}
                previewUrl={previewUrl}
                explanation={session.explanation}
                onExplanation={setExplanation}
              />
            </div>

            {/* ---- Metrics tab ---- */}
            {activeTab === "metrics" && <MetricsTab result={result} />}
          </section>
        )}
      </main>
    </div>
  );
}
