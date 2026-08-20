"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAuth } from "@/context/AuthContext";
import { getStats, type StatsData } from "@/lib/api";
import OperationalCharts from "@/components/OperationalCharts";
import Header from "@/components/Header";
import styles from "./page.module.css";

/**
 * Monitoring dashboard page. Protected — redirects to /login if not
 * authenticated.
 *
 * Business Insights section: confidence distribution histogram, prediction
 * volume over time, landmark distribution, feedback breakdown,
 * per-model-version stats.
 * Operational Monitoring section: serving model version, Prometheus flag,
 * and an honest pending note for request volume / latency / error-rate.
 */
export default function DashboardPage() {
  const router = useRouter();
  const { user, loading: authLoading, logout } = useAuth();

  const [stats, setStats] = useState<StatsData | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);

  /* ---------- auth gate ---------- */
  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  /* ---------- fetch stats ---------- */
  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    async function load() {
      setStatsLoading(true);
      const { data, error, status } = await getStats();
      if (cancelled) return;
      if (status === 401) {
        await logout();
        router.replace("/login");
        return;
      }
      if (error || !data) {
        setStatsError(error ?? "Failed to load statistics.");
      } else {
        setStats(data);
      }
      setStatsLoading(false);
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [user, logout, router]);

  if (authLoading || !user) {
    return (
      <div className={styles.loadingOverlay}>
        <span className="spinner" />
      </div>
    );
  }

  const biz = stats?.business;
  const ops = stats?.operational;

  return (
    <div className={styles.page}>
      <Header active="dashboard" />

      <main className={styles.main}>
        {/* page heading */}
        <div>
          <h1 className={styles.pageTitle}>Monitoring Dashboard</h1>
          <p className={styles.pageSubtitle}>
            Aggregated statistics from the predictions and feedback tables.
          </p>
        </div>

        {statsLoading && (
          <div style={{ display: "flex", justifyContent: "center", padding: "var(--space-8)" }}>
            <span className="spinner" />
          </div>
        )}

        {statsError && (
          <div className={`alert alert-error ${styles.errorBox}`}>{statsError}</div>
        )}

        {stats && (
          <>
            {/* ══ Business Insights ══════════════════════════════════════ */}
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>Business Insights</h2>

              {/* KPI row */}
              <div className={styles.kpiRow}>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Total Predictions</div>
                  <div className={styles.kpiValue}>
                    {biz!.total_predictions.toLocaleString()}
                  </div>
                </div>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Avg Confidence</div>
                  <div className={styles.kpiValue}>
                    {biz!.avg_confidence != null
                      ? biz!.avg_confidence.toFixed(3)
                      : "—"}
                  </div>
                </div>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Feedback Bad-Rate</div>
                  <div className={styles.kpiValue}>
                    {biz!.feedback.total > 0
                      ? `${(biz!.feedback.bad_rate * 100).toFixed(1)}%`
                      : "—"}
                  </div>
                </div>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Model Versions</div>
                  <div className={styles.kpiValue}>
                    {biz!.per_model_version.length}
                  </div>
                </div>
              </div>

              {/* Charts row */}
              <div className={styles.chartRow}>
                {/* Confidence histogram */}
                <div className={`card ${styles.chartCard}`}>
                  <div className={styles.chartTitle}>
                    Confidence Distribution
                  </div>
                  {biz!.total_predictions === 0 ? (
                    <div className={styles.emptyChart}>
                      No predictions yet.
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart
                        data={biz!.confidence_histogram}
                        margin={{ top: 0, right: 8, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="bucket_label"
                          tick={{ fontSize: 10 }}
                          tickLine={false}
                        />
                        <YAxis tick={{ fontSize: 10 }} tickLine={false} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#6366f1" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </div>

                {/* Prediction volume over time */}
                <div className={`card ${styles.chartCard}`}>
                  <div className={styles.chartTitle}>
                    Prediction Volume (30 days)
                  </div>
                  {biz!.predictions_over_time.length === 0 ? (
                    <div className={styles.emptyChart}>
                      No predictions in the last 30 days.
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart
                        data={biz!.predictions_over_time}
                        margin={{ top: 0, right: 8, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 10 }}
                          tickLine={false}
                          tickFormatter={(v: string) => v.slice(5)}
                        />
                        <YAxis tick={{ fontSize: 10 }} tickLine={false} />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey="count"
                          stroke="#6366f1"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              {/* Landmark count distribution */}
              <div className={`card ${styles.chartCard}`}>
                <div className={styles.chartTitle}>
                  Landmark Count Distribution
                </div>
                {biz!.landmark_distribution.length === 0 ? (
                  <div className={styles.emptyChart}>
                    No landmark data yet.
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart
                      data={biz!.landmark_distribution}
                      margin={{ top: 0, right: 8, left: -20, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="landmark_count"
                        tick={{ fontSize: 10 }}
                        tickLine={false}
                        label={{ value: "Landmarks detected", position: "insideBottom", offset: -2, fontSize: 10 }}
                      />
                      <YAxis tick={{ fontSize: 10 }} tickLine={false} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#10b981" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>

              {/* Feedback breakdown */}
              <div className={`card ${styles.chartCard}`}>
                <div className={styles.chartTitle}>Feedback Breakdown</div>
                {biz!.feedback.total === 0 ? (
                  <div className={styles.emptyChart}>
                    No feedback submitted yet.
                  </div>
                ) : (
                  <div className={styles.feedbackRow}>
                    <div
                      className={`${styles.feedbackChip} ${styles.feedbackChipGood}`}
                    >
                      <span className={styles.feedbackChipCount}>
                        {biz!.feedback.good}
                      </span>
                      <span className={styles.feedbackChipLabel}>Good</span>
                    </div>
                    <div
                      className={`${styles.feedbackChip} ${styles.feedbackChipBad}`}
                    >
                      <span className={styles.feedbackChipCount}>
                        {biz!.feedback.bad}
                      </span>
                      <span className={styles.feedbackChipLabel}>Bad</span>
                    </div>
                    <div
                      className={`${styles.feedbackChip} ${styles.feedbackChipUncertain}`}
                    >
                      <span className={styles.feedbackChipCount}>
                        {biz!.feedback.uncertain}
                      </span>
                      <span className={styles.feedbackChipLabel}>Uncertain</span>
                    </div>
                    <div className={styles.feedbackChip}>
                      <span className={styles.feedbackChipCount}>
                        {(biz!.feedback.bad_rate * 100).toFixed(1)}%
                      </span>
                      <span className={styles.feedbackChipLabel}>Bad Rate</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Per-model-version table */}
              {biz!.per_model_version.length > 0 && (
                <div className={`card ${styles.chartCard}`}>
                  <div className={styles.chartTitle}>
                    Per-Model-Version Statistics
                  </div>
                  <div className="table-container">
                    <table className={styles.table}>
                      <thead>
                        <tr>
                          <th>Model Version</th>
                          <th>Predictions</th>
                          <th>Avg Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {biz!.per_model_version.map((mv) => (
                          <tr key={mv.model_version}>
                            <td>{mv.model_version}</td>
                            <td>{mv.prediction_count}</td>
                            <td>
                              {mv.avg_confidence != null
                                ? mv.avg_confidence.toFixed(3)
                                : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </section>

            <hr className="divider" />

            {/* ══ Operational Monitoring ═════════════════════════════════ */}
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>Operational Monitoring</h2>

              {/* KPI row — serving version, prometheus flag, request scalars */}
              <div className={styles.kpiRow}>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Serving Model Version</div>
                  <div className={styles.kpiValue}>
                    {ops!.serving_model_version ?? "—"}
                  </div>
                </div>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Prometheus /metrics</div>
                  <div className={styles.kpiValue}>
                    {ops!.prometheus_metrics_available ? (
                      <span className={styles.badgePill}>✓ Available</span>
                    ) : (
                      <span>Not available</span>
                    )}
                  </div>
                </div>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Total Requests</div>
                  <div className={styles.kpiValue}>
                    {ops!.request_metrics.total_requests > 0
                      ? ops!.request_metrics.total_requests.toLocaleString()
                      : "—"}
                  </div>
                </div>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Error Rate</div>
                  <div className={styles.kpiValue}>
                    {ops!.request_metrics.total_requests > 0
                      ? `${(ops!.request_metrics.error_rate * 100).toFixed(1)}%`
                      : "—"}
                  </div>
                </div>
              </div>

              <div className={styles.kpiRow}>
                <div className={`card ${styles.kpiCard}`}>
                  <div className={styles.kpiLabel}>Avg Latency (ms)</div>
                  <div className={styles.kpiValue}>
                    {ops!.request_metrics.avg_latency_ms != null
                      ? ops!.request_metrics.avg_latency_ms.toFixed(1)
                      : "—"}
                  </div>
                </div>
              </div>

              <p className={styles.noteBox}>
                The KPIs above are point-in-time scalars from the backend&apos;s
                in-process registry (one worker). The charts below are the full
                time-series, scraped and aggregated by Prometheus.
              </p>

              {/* Live time-series from Prometheus: request rate, latency
                  percentiles and error rate. Renders a "not configured" note
                  where PROMETHEUS_URL is unset, so it is safe in every
                  environment (the KPIs above work without Prometheus). */}
              <OperationalCharts canRunDriftCheck={user.role === "admin"} />
            </section>
          </>
        )}
      </main>
    </div>
  );
}
