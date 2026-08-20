/**
 * Shared confidence thresholds and helpers.
 *
 * These thresholds mirror the cv-pipeline spec (task_328 section 13) and the
 * backend monitoring alert rules, so the frontend tells the SAME story as the
 * Prometheus / Azure Monitor metrics: a viewer reading the UI sees the same
 * "good / watch / alert" boundaries the ops dashboards use.
 */

/** Sigmoid midpoint: below this the model is effectively uncertain. */
export const CONF_UNCERTAIN = 0.5;

/** ALERT_CONFIDENCE_MIN default: mean confidence below this raises an alert. */
export const CONF_ALERT = 0.6;

/** Map a confidence in [0, 1] to a themed colour variable. */
export function confidenceColor(c: number): string {
  if (c >= 0.8) return "var(--color-success)";
  if (c >= CONF_ALERT) return "var(--color-warning)";
  return "var(--color-danger)";
}

/** Map a confidence in [0, 1] to a short human label. */
export function confidenceLabel(c: number): string {
  if (c >= 0.8) return "High";
  if (c >= CONF_ALERT) return "Moderate";
  if (c >= CONF_UNCERTAIN) return "Low";
  return "Very low";
}
