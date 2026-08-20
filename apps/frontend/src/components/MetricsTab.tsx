"use client";

import { type InferenceResult } from "@/lib/api";
import {
  CONF_ALERT,
  CONF_UNCERTAIN,
  confidenceColor,
  confidenceLabel,
} from "@/lib/confidence";
import styles from "./MetricsTab.module.css";

/**
 * Stakeholder-facing metrics view for a single prediction.
 *
 * Deliberately plots ``mask_confidence`` on the same 0–1 scale and against the
 * same threshold lines (0.50 uncertain, 0.60 alert) that the backend exports
 * as the ``cv_inference_confidence`` Prometheus histogram and uses in its
 * Azure Monitor alert rules — so this view and the ops dashboards agree.
 */
export default function MetricsTab({ result }: { result: InferenceResult }) {
  const conf = result.mask_confidence;
  const confColour = confidenceColor(conf);

  return (
    <div className={styles.wrap}>
      <div className={styles.cards}>
        <Card label="Landmarks Detected" value={String(result.landmark_count)} />
        <Card
          label="Mask Confidence"
          value={conf.toFixed(3)}
          sub={confidenceLabel(conf)}
          colour={confColour}
        />
        <Card label="Model Version" value={result.model_version} />
        <Card label="Pipeline Version" value={result.pipeline_version} />
      </div>

      {/* confidence scale aligned to the monitoring thresholds */}
      <div className={`card ${styles.panel}`}>
        <div className={styles.panelTitle}>Mask confidence vs alert thresholds</div>
        <div className={styles.scale}>
          <div
            className={styles.scaleFill}
            style={{ width: `${conf * 100}%`, background: confColour }}
          />
          <Threshold pos={CONF_UNCERTAIN} label="uncertain" />
          <Threshold pos={CONF_ALERT} label="alert" />
        </div>
        <div className={styles.axis}>
          <span>0.0</span>
          <span>0.5</span>
          <span>1.0</span>
        </div>
        <p className={styles.note}>
          Below {CONF_ALERT.toFixed(2)} sustained over an hour raises a drift
          alert; {CONF_UNCERTAIN.toFixed(2)} is the model&apos;s decision
          midpoint. Same thresholds as the backend monitoring.
        </p>
      </div>

      {/* per-landmark confidence */}
      {result.landmarks.length > 0 && (
        <div className={`card ${styles.panel}`}>
          <div className={styles.panelTitle}>Per-landmark confidence</div>
          {result.landmarks.map((lm) => (
            <div key={lm.id} className={styles.lmRow}>
              <span className={styles.lmId}>Tip {lm.id}</span>
              <div className={styles.lmTrack}>
                <div
                  className={styles.lmFill}
                  style={{
                    width: `${lm.confidence * 100}%`,
                    background: confidenceColor(lm.confidence),
                  }}
                />
              </div>
              <span className={styles.lmVal}>{lm.confidence.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Card({
  label,
  value,
  sub,
  colour,
}: {
  label: string;
  value: string;
  sub?: string;
  colour?: string;
}) {
  return (
    <div className={`card ${styles.metricCard}`}>
      <div className={styles.metricLabel}>{label}</div>
      <div className={styles.metricValue} style={colour ? { color: colour } : undefined}>
        {value}
      </div>
      {sub && <div className={styles.metricSub}>{sub}</div>}
    </div>
  );
}

function Threshold({ pos, label }: { pos: number; label: string }) {
  return (
    <div className={styles.threshold} style={{ left: `${pos * 100}%` }}>
      <span className={styles.thresholdLabel}>{label}</span>
    </div>
  );
}
