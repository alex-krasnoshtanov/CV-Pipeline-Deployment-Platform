"use client";

import { useCallback, useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getMetricsRange,
  postMonitoringCheck,
  type PromRangeResponse,
} from "@/lib/api";
import styles from "./OperationalCharts.module.css";

/**
 * Live operational metrics, read from Prometheus through the
 * /api/prometheus/query_range proxy.
 *
 * These are the time-series the point-in-time KPIs above cannot show: request
 * rate, latency percentiles and error rate over a chosen window. The same
 * component runs in every environment - it just renders a "not configured"
 * note where PROMETHEUS_URL is unset (e.g. before the monitoring stack is
 * deployed), so the dashboard never breaks on a missing Prometheus.
 *
 * Charting note: only the recharts primitives the dashboard already uses are
 * imported here (no Legend/Area), so one shared recharts mock covers the whole
 * page in tests. The latency legend is therefore drawn with plain elements.
 */

// Each window pairs a lookback with a sensible step (chart resolution) and a
// rate() window. Wider windows use a coarser step to cap the number of points
// and a longer rate() window to keep lines readable rather than spiky.
const RANGES = {
  "15m": { label: "15m", minutes: 15, stepSeconds: 15, rate: "1m" },
  "1h": { label: "1h", minutes: 60, stepSeconds: 60, rate: "5m" },
  "6h": { label: "6h", minutes: 360, stepSeconds: 300, rate: "15m" },
} as const;

type RangeKey = keyof typeof RANGES;

// Poll at the scrape interval: new data lands in Prometheus every 15s, so
// refreshing faster would just redraw identical points.
const REFRESH_MS = 15_000;

const CHART_MARGIN = { top: 8, right: 12, left: -12, bottom: 0 };

// Alert threshold for the low-confidence fraction, drawn as a dashed reference
// line and used for the at-a-glance badge. Mirrors the backend default
// ALERT_LOW_CONF_FRACTION (0.20); it is a display constant only, so if the
// backend value is overridden via env, update this to keep the line in sync.
const ALERT_FRACTION_THRESHOLD = 0.2;

type Row = { t: number } & Record<string, number>;

/**
 * Merge several single-series range responses into one array of rows keyed by
 * timestamp, e.g. [{ t, p50, p95, p99 }, ...]. Each response is expected to
 * carry one series (our queries aggregate with sum()/histogram_quantile()).
 */
function mergeRangeResponses(
  entries: { key: string; resp: PromRangeResponse }[],
): Row[] {
  const byTime = new Map<number, Row>();
  for (const { key, resp } of entries) {
    const series = resp.data?.result?.[0];
    if (!series) continue;
    for (const [ts, valueStr] of series.values) {
      const value = Number(valueStr);
      if (!Number.isFinite(value)) continue; // skip NaN (e.g. 0/0 error rate)
      const row = byTime.get(ts) ?? ({ t: ts } as Row);
      row[key] = value;
      byTime.set(ts, row);
    }
  }
  return [...byTime.values()].sort((a, b) => a.t - b.t);
}

/** Format a unix-seconds timestamp as HH:MM for the x-axis. */
function hhmm(ts: number): string {
  const d = new Date(ts * 1000);
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(d.getMinutes()).padStart(2, "0");
  return `${h}:${m}`;
}

/**
 * Live operational charts (request rate, error rate, latency percentiles, and
 * the model low-confidence fraction), read from Prometheus through the
 * /api/prometheus/query_range proxy. Renders a "not configured" note where
 * PROMETHEUS_URL is unset, so it is safe in every environment.
 *
 * @param props - Component props.
 * @param props.canRunDriftCheck - When true, shows an admin-only button that
 *   POSTs /api/monitoring/check and then refreshes the drift gauges.
 * @returns The operational charts panel.
 */
export default function OperationalCharts({
  canRunDriftCheck = false,
}: {
  canRunDriftCheck?: boolean;
}) {
  const [range, setRange] = useState<RangeKey>("1h");
  const [status, setStatus] = useState<
    "loading" | "success" | "disabled" | "error"
  >("loading");
  const [error, setError] = useState<string | null>(null);
  const [rateData, setRateData] = useState<Row[]>([]);
  const [latencyData, setLatencyData] = useState<Row[]>([]);
  const [errorData, setErrorData] = useState<Row[]>([]);
  const [driftData, setDriftData] = useState<Row[]>([]);
  const [driftLatest, setDriftLatest] = useState<number | null>(null);
  const [alertState, setAlertState] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [runMsg, setRunMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    const cfg = RANGES[range];
    const opts = { minutes: cfg.minutes, stepSeconds: cfg.stepSeconds };
    const w = cfg.rate;

    const reqTotal = `sum(rate(http_requests_total[${w}]))`;
    const errRate =
      `sum(rate(http_requests_total{status=~"4xx|5xx"}[${w}]))` +
      ` / sum(rate(http_requests_total[${w}])) * 100`;
    // p* latency from the high-resolution histogram (no handler label = the
    // most accurate overall percentile). * 1000 converts seconds to ms.
    const pq = (q: number) =>
      `histogram_quantile(${q}, ` +
      `sum(rate(http_request_duration_highr_seconds_bucket[${w}])) by (le))` +
      ` * 1000`;

    // Drift signal: the low-confidence fraction is a gauge (queried directly,
    // no rate()); the alert gauge is the backend's own 0/1 threshold decision.
    const driftFracQ = "cv_low_confidence_fraction";
    const driftAlertQ = "cv_low_confidence_alert";

    const [rate, p50, p95, p99, err, fracResp, alertResp] = await Promise.all([
      getMetricsRange(reqTotal, opts),
      getMetricsRange(pq(0.5), opts),
      getMetricsRange(pq(0.95), opts),
      getMetricsRange(pq(0.99), opts),
      getMetricsRange(errRate, opts),
      getMetricsRange(driftFracQ, opts),
      getMetricsRange(driftAlertQ, opts),
    ]);

    // A "disabled" response means Prometheus is not wired up in this env at all.
    if (rate.status === "disabled") {
      setStatus("disabled");
      return;
    }
    // Only treat it as a hard error if everything failed; a single failing
    // query should still let the other charts render.
    if (
      rate.status === "error" &&
      p95.status === "error" &&
      err.status === "error"
    ) {
      setStatus("error");
      setError(rate.error ?? "Could not load metrics.");
      return;
    }

    setRateData(mergeRangeResponses([{ key: "rate", resp: rate }]));
    setLatencyData(
      mergeRangeResponses([
        { key: "p50", resp: p50 },
        { key: "p95", resp: p95 },
        { key: "p99", resp: p99 },
      ]),
    );
    setErrorData(mergeRangeResponses([{ key: "err", resp: err }]));

    // Stamp a constant threshold column on each drift row so the dashed
    // reference line spans the chart using a plain Line - no recharts
    // ReferenceLine, so the shared recharts test mock stays unchanged.
    const driftRows = mergeRangeResponses([
      { key: "frac", resp: fracResp },
    ]).map((row): Row => ({ ...row, threshold: ALERT_FRACTION_THRESHOLD }));
    setDriftData(driftRows);
    setDriftLatest(
      driftRows.length ? driftRows[driftRows.length - 1].frac : null,
    );
    const alertSeries = alertResp.data?.result?.[0];
    setAlertState(
      alertSeries?.values?.length
        ? Number(alertSeries.values[alertSeries.values.length - 1][1])
        : null,
    );

    setStatus("success");
    setError(null);
  }, [range]);

  useEffect(() => {
    // Initial fetch, then poll at the scrape interval. The fetch is wrapped in a
    // locally-defined async function (rather than calling the memoized load
    // directly) so its state updates are deferred, not run synchronously in the
    // effect body. Old data stays on screen between refreshes, so the spinner
    // only shows on the first load (while status is still its initial "loading").
    const run = async () => {
      await load();
    };
    run();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  // Admin-only: trigger the backend drift check on demand, then refresh the
  // gauges so the chart and badge reflect the new computation immediately.
  async function runCheck() {
    setRunning(true);
    setRunMsg(null);
    const { data, error: postError } = await postMonitoringCheck();
    setRunning(false);
    if (postError) {
      setRunMsg(postError);
      return;
    }
    const drift = data?.drift;
    if (drift?.status === "no_data") {
      setRunMsg("No predictions yet to evaluate.");
    } else if (drift?.alert) {
      setRunMsg("Drift detected: low confidence is above threshold.");
    } else {
      setRunMsg("Check complete: within threshold.");
    }
    await load();
  }

  if (status === "disabled") {
    return (
      <div className={`card ${styles.note}`}>
        <strong>Live metrics are not configured in this environment.</strong>
        <p>
          The operational time-series come from Prometheus. Set
          <code> PROMETHEUS_URL</code> on the frontend (pointing at the
          Prometheus service) to enable them. The point-in-time figures above
          come straight from the backend and work without Prometheus.
        </p>
      </div>
    );
  }

  if (status === "loading") {
    return (
      <div className={styles.center}>
        <span className="spinner" />
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className={`alert alert-error ${styles.note}`}>
        {error ?? "Could not load live metrics."}
      </div>
    );
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.toolbar}>
        <span className={styles.live}>
          <span className={styles.liveDot} /> Live
        </span>
        <div className={styles.toolbarRight}>
          <div className={styles.ranges}>
            {(Object.keys(RANGES) as RangeKey[]).map((k) => (
              <button
                key={k}
                className={`btn btn-ghost ${range === k ? styles.rangeActive : ""}`}
                onClick={() => setRange(k)}
              >
                {RANGES[k].label}
              </button>
            ))}
          </div>
          {canRunDriftCheck && (
            <>
              <button
                id="btn-run-drift"
                className="btn btn-secondary"
                onClick={runCheck}
                disabled={running}
              >
                {running ? (
                  <>
                    <span className="spinner spinner-sm" /> Checking...
                  </>
                ) : (
                  "Run drift check"
                )}
              </button>
              {runMsg && <span className={styles.runMsg}>{runMsg}</span>}
            </>
          )}
        </div>
      </div>

      <div className={styles.grid}>
        <ChartCard title="Request Rate (req/s)" empty={rateData.length === 0}>
          <LineChart data={rateData} margin={CHART_MARGIN}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fontSize: 10 }}
              tickLine={false}
              tickFormatter={hhmm}
              minTickGap={32}
            />
            <YAxis tick={{ fontSize: 10 }} tickLine={false} width={36} />
            <Tooltip
              labelFormatter={(label) => hhmm(Number(label))}
              formatter={(value) => [Number(value).toFixed(2), "req/s"]}
            />
            <Line
              type="monotone"
              dataKey="rate"
              stroke="#6366f1"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartCard>

        <ChartCard title="Error Rate (%)" empty={errorData.length === 0}>
          <LineChart data={errorData} margin={CHART_MARGIN}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fontSize: 10 }}
              tickLine={false}
              tickFormatter={hhmm}
              minTickGap={32}
            />
            <YAxis
              tick={{ fontSize: 10 }}
              tickLine={false}
              width={36}
              domain={[0, "auto"]}
            />
            <Tooltip
              labelFormatter={(label) => hhmm(Number(label))}
              formatter={(value) => [`${Number(value).toFixed(1)}%`, "errors"]}
            />
            <Line
              type="monotone"
              dataKey="err"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartCard>

        <ChartCard
          title="Latency (ms)"
          empty={latencyData.length === 0}
          className={styles.wide}
          legend={
            <div className={styles.legend}>
              <LegendDot colour="#10b981" label="p50" />
              <LegendDot colour="#6366f1" label="p95" />
              <LegendDot colour="#f59e0b" label="p99" />
            </div>
          }
        >
          <LineChart data={latencyData} margin={CHART_MARGIN}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fontSize: 10 }}
              tickLine={false}
              tickFormatter={hhmm}
              minTickGap={32}
            />
            <YAxis tick={{ fontSize: 10 }} tickLine={false} width={44} />
            <Tooltip
              labelFormatter={(label) => hhmm(Number(label))}
              formatter={(value, name) => [`${Number(value).toFixed(0)} ms`, name]}
            />
            <Line type="monotone" dataKey="p50" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
            <Line type="monotone" dataKey="p95" stroke="#6366f1" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
            <Line type="monotone" dataKey="p99" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
          </LineChart>
        </ChartCard>

        <ChartCard
          title="Low-confidence Fraction"
          empty={driftData.length === 0}
          className={styles.wide}
          legend={
            <span className={styles.driftHead}>
              <DriftBadge alert={alertState} />
              <span className={styles.badgeValue}>
                {driftLatest != null
                  ? `${(driftLatest * 100).toFixed(1)}% now`
                  : "no data"}
              </span>
            </span>
          }
        >
          <LineChart data={driftData} margin={CHART_MARGIN}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="t"
              tick={{ fontSize: 10 }}
              tickLine={false}
              tickFormatter={hhmm}
              minTickGap={32}
            />
            <YAxis
              tick={{ fontSize: 10 }}
              tickLine={false}
              width={44}
              domain={[0, 1]}
              tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`}
            />
            <Tooltip
              labelFormatter={(label) => hhmm(Number(label))}
              formatter={(value, name) => [
                `${(Number(value) * 100).toFixed(1)}%`,
                name === "threshold" ? "alert threshold" : "low-conf fraction",
              ]}
            />
            <Line
              type="monotone"
              dataKey="frac"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="threshold"
              stroke="#9ca3af"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartCard>
      </div>
    </div>
  );
}

function ChartCard({
  title,
  empty,
  children,
  className,
  legend,
}: {
  title: string;
  empty: boolean;
  children: React.ReactElement;
  className?: string;
  legend?: React.ReactNode;
}) {
  return (
    <div className={`card ${styles.chartCard} ${className ?? ""}`}>
      <div className={styles.chartHead}>
        <span className={styles.chartTitle}>{title}</span>
        {legend}
      </div>
      {empty ? (
        <div className={styles.empty}>No data in this window yet.</div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          {children}
        </ResponsiveContainer>
      )}
    </div>
  );
}

function LegendDot({ colour, label }: { colour: string; label: string }) {
  return (
    <span className={styles.legendItem}>
      <span className={styles.dot} style={{ background: colour }} />
      {label}
    </span>
  );
}

function DriftBadge({ alert }: { alert: number | null }) {
  if (alert == null) {
    return (
      <span className={`${styles.badge} ${styles.badgeNeutral}`}>
        No data yet
      </span>
    );
  }
  if (alert >= 1) {
    return (
      <span className={`${styles.badge} ${styles.badgeAlert}`}>Drift alert</span>
    );
  }
  return (
    <span className={`${styles.badge} ${styles.badgeOk}`}>Within threshold</span>
  );
}
