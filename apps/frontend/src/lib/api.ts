/**
 * API client for the FastAPI backend.
 *
 * All requests go through the Next.js proxy at /api/*, which forwards
 * to the backend server-side. This keeps the backend URL out of the
 * browser bundle entirely — no NEXT_PUBLIC_ env var needed.
 *
 * Cookies are same-origin (browser <-> Next.js), so `credentials:
 * "include"` works without CORS configuration.
 */

const API_PREFIX = "/api";

/* ---------- types ---------- */

/** Authenticated user profile returned by the backend. */
export interface User {
  id: string;
  name: string;
  role: string;
  email: string | null;
  auth_method?: "oauth" | "password" | "api_key" | null;
}

/** Backend health-check response. */
export interface HealthData {
  status: "ok" | "loading";
  model_loaded: boolean;
  model_version: string | null;
  pipeline_version: string;
  serving_mode: "local" | "azure_ml" | null;
  timestamp: string;
}

/** A single detected landmark with pixel coordinates and confidence. */
export interface Landmark {
  id: number;
  x: number;
  y: number;
  confidence: number;
}

/** Full inference response including mask, metrics, and landmark list. */
export interface InferenceResult {
  pipeline_version: string;
  model_version: string;
  timestamp: string;
  image_filename: string;
  image_width_px: number;
  image_height_px: number;
  metadata: {
    plate_id: string | null;
    experiment_id: string | null;
    timestamp: string | null;
  };
  mask_b64: string;
  mask_confidence: number;
  landmark_count: number;
  landmarks: Landmark[];
  prediction_id: string | null;
}

/** Confirmation payload returned after submitting feedback. */
export interface FeedbackResult {
  feedback_id: string;
  prediction_id: string;
  flag: string;
  created_at: string;
}

/** Shape of error responses from the backend. */
export interface ApiError {
  error_code?: string;
  message?: string;
  detail?:
    | { error_code?: string; message?: string }
    | Array<{ msg?: string }>;
}

/* ---------- helpers ---------- */

function parseError(body: ApiError | null): string {
  if (!body) return "Cannot reach the backend. Is it running?";

  const codeMessages: Record<string, string> = {
    EMAIL_TAKEN: "An account with that email already exists.",
    INVALID_CREDENTIALS: "Email or password is incorrect.",
    UNAUTHORIZED: "You need to sign in to do that.",
    FORBIDDEN: "Your account does not have permission for this action.",
    NOT_FOUND: "The requested resource was not found.",
    VALIDATION_ERROR: "The request was invalid. Check your input.",
    MISCONFIGURED: "The server is misconfigured. Contact an admin.",
    RATE_LIMITED: "Too many requests. Please wait a moment and try again.",
  };

  const code = body.error_code;
  if (code && codeMessages[code]) return codeMessages[code];

  const detail = body.detail;
  if (detail && !Array.isArray(detail) && detail.message) {
    return detail.message;
  }
  if (detail && !Array.isArray(detail) && detail.error_code) {
    const dc = detail.error_code;
    if (dc && codeMessages[dc]) return codeMessages[dc];
  }
  if (Array.isArray(detail)) {
    const msgs = detail
      .filter((d): d is { msg: string } => typeof d.msg === "string")
      .map((d) => d.msg);
    if (msgs.length) return "Validation failed: " + msgs.join("; ");
  }

  if (body.message) return body.message;

  return "An unexpected error occurred.";
}

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<{ data: T | null; error: string | null; status: number }> {
  try {
    const res = await fetch(`${API_PREFIX}${path}`, {
      credentials: "include",
      ...init,
    });
    if (res.status === 204) {
      return { data: null, error: null, status: 204 };
    }
    const body = await res.json().catch(() => null);
    if (res.ok) {
      return { data: body as T, error: null, status: res.status };
    }
    // A null body on a gateway status means a non-JSON error page (proxy or
    // ingress timeout/unavailable), not "backend down". Explanations can be
    // slow on a cold start, so say so instead of the misleading network error.
    if (body === null && (res.status === 502 || res.status === 503 || res.status === 504)) {
      return {
        data: null,
        error:
          "The server took too long or is temporarily unavailable. " +
          "Explanations can be slow on a cold start \u2014 please try again.",
        status: res.status,
      };
    }
    return { data: null, error: parseError(body), status: res.status };
  } catch {
    return {
      data: null,
      error: "Cannot reach the backend. Is it running?",
      status: 0,
    };
  }
}

/* ---------- auth ---------- */

/** Authenticate a user with email and password. Sets a session cookie. */
export function login(email: string, password: string) {
  return request<User>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

/** Create a new user account and log in automatically. */
export function register(name: string, email: string, password: string) {
  return request<User>("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
}

/** End the current session and clear the session cookie. */
export function logout() {
  return request<void>("/auth/logout", { method: "POST" });
}

/** Fetch the currently authenticated user's profile via session cookie. */
export function getMe() {
  return request<User>("/auth/me");
}

/* ---------- operational ---------- */

/** Poll the backend health endpoint for model and pipeline status. */
export function getHealth() {
  return request<HealthData>("/health");
}

/** Submit an image for inference. Expects a multipart form with an `image` field. */
export async function postInfer(formData: FormData) {
  return request<InferenceResult>("/infer", {
    method: "POST",
    body: formData,
    // Do NOT set Content-Type — browser sets multipart boundary automatically.
  });
}

/** Submit user feedback for a prediction. */
export function postFeedback(payload: {
  prediction_id: string;
  flag: string;
  notes?: string;
}) {
  return request<FeedbackResult>("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/* ---------- monitoring ---------- */

/** Drift sub-object returned by POST /monitoring/check. */
export interface DriftCheckResult {
  /** Present (e.g. "no_data") only when there were no predictions to score. */
  status?: string;
  /** Number of recent predictions sampled. */
  window?: number;
  /** Fraction of those below the confidence floor. */
  low_confidence_fraction?: number;
  /** Whether that fraction crossed the alert threshold. */
  alert?: boolean;
}

/** Response body of POST /monitoring/check. */
export interface MonitoringCheckResponse {
  drift: DriftCheckResult;
}

/**
 * Trigger the backend rolling-confidence drift check (admin only).
 *
 * The backend reads recent predictions, updates the Prometheus drift gauges,
 * and returns the computed fraction/alert. Never throws: a non-admin caller
 * comes back with `error` set and `status` 403.
 */
export function postMonitoringCheck() {
  return request<MonitoringCheckResponse>("/monitoring/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
}


/* ---------- explainability ---------- */

/** Seg-Grad-CAM explanation returned by POST /explain. */
export interface ExplanationResult {
  pipeline_version: string;
  model_version: string;
  timestamp: string;
  image_filename: string;
  image_width_px: number;
  image_height_px: number;
  metadata: {
    plate_id: string | null;
    experiment_id: string | null;
    timestamp: string | null;
  };
  method: string;
  target_layer: string;
  downscaled: boolean;
  heatmap_peak: number;
  heatmap_b64: string;
}

/** Request a Grad-CAM explanation heatmap for an image (multipart form). */
export async function postExplain(formData: FormData) {
  return request<ExplanationResult>("/explain", {
    method: "POST",
    body: formData,
    // Do NOT set Content-Type — browser sets the multipart boundary.
  });
}

/* ---------- stats ---------- */

/** One bar in the confidence-score histogram returned by GET /stats. */
export interface ConfidenceBucket {
  bucket_label: string;
  count: number;
}

/** Prediction count for one calendar day in the time-series chart. */
export interface TimePoint {
  date: string;
  count: number;
}

/** Aggregate feedback flag counts and derived bad-rate. */
export interface FeedbackBreakdown {
  good: number;
  bad: number;
  uncertain: number;
  total: number;
  bad_rate: number;
}

/** Prediction count grouped by landmark_count value. */
export interface LandmarkBucket {
  landmark_count: number;
  count: number;
}

/** Per-model-version prediction volume and average confidence. */
export interface ModelVersionStat {
  model_version: string;
  prediction_count: number;
  avg_confidence: number | null;
}

/** Business-insight metrics from the predictions and feedback tables. */
export interface BusinessStats {
  total_predictions: number;
  avg_confidence: number | null;
  confidence_histogram: ConfidenceBucket[];
  predictions_over_time: TimePoint[];
  landmark_distribution: LandmarkBucket[];
  feedback: FeedbackBreakdown;
  per_model_version: ModelVersionStat[];
}

/** HTTP request counters and latency sourced from the Prometheus registry. */
export interface RequestMetrics {
  total_requests: number;
  error_rate: number; // fraction 0.0–1.0
  avg_latency_ms: number | null; // null when no traffic yet
}

/** Operational monitoring metrics from GET /stats. */
export interface OperationalStats {
  serving_model_version: string | null;
  prometheus_metrics_available: boolean;
  request_metrics: RequestMetrics;
}

/** Top-level response from GET /stats. */
export interface StatsData {
  business: BusinessStats;
  operational: OperationalStats;
}

/** Fetch aggregated monitoring dashboard statistics. */
export function getStats() {
  return request<StatsData>("/stats");
}


/* ---------- operational metrics (Prometheus) ---------- */

/** One [unixSeconds, value] sample in a Prometheus range result. */
export type PromSample = [number, string];

/** One time series (a metric with its label set) in a range result. */
export interface PromSeries {
  metric: Record<string, string>;
  values: PromSample[];
}

/**
 * Result of a Prometheus range query, as returned by the
 * /api/prometheus/query_range proxy.
 *
 * `status` is a discriminator the dashboard branches on:
 * - "success":  `data.result` holds the series.
 * - "disabled": PROMETHEUS_URL is not set in this environment.
 * - "error":    bad query, or Prometheus unreachable.
 */
export interface PromRangeResponse {
  status: "success" | "disabled" | "error";
  data?: { resultType: string; result: PromSeries[] };
  error?: string;
}

/**
 * Run a Prometheus range query through the server-side proxy.
 *
 * The caller gives a window (`minutes` back from now) and a `stepSeconds`
 * resolution; the absolute start/end are computed here so every chart on the
 * page shares one consistent time axis. Returns a `status: "error"` object
 * (never throws) so a single failing tile cannot crash the dashboard.
 */
export async function getMetricsRange(
  query: string,
  opts: { minutes: number; stepSeconds: number },
): Promise<PromRangeResponse> {
  const end = Math.floor(Date.now() / 1000);
  const start = end - opts.minutes * 60;
  const qs =
    `query=${encodeURIComponent(query)}` +
    `&start=${start}&end=${end}&step=${opts.stepSeconds}`;
  try {
    const res = await fetch(`${API_PREFIX}/prometheus/query_range?${qs}`, {
      credentials: "include",
    });
    const body = (await res
      .json()
      .catch(() => null)) as PromRangeResponse | null;
    if (!body || !body.status) {
      return {
        status: "error",
        error: "Unexpected response from metrics proxy.",
      };
    }
    return body;
  } catch {
    return { status: "error", error: "Cannot reach the metrics proxy." };
  }
}
