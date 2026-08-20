/**
 * Server-side proxy for Prometheus range queries.
 *
 * Mirrors the backend proxy in app/api/[...path]/route.ts, for the same two
 * reasons:
 *
 * 1. PROMETHEUS_URL is a plain server-side env var (no NEXT_PUBLIC_ prefix), so
 *    it is read at runtime, not inlined at build time. The same frontend image
 *    therefore works in local, on-prem and cloud with only the env var changed.
 *
 * 2. Prometheus is never exposed to the browser. The browser only ever talks to
 *    the Next.js origin, which keeps Prometheus on the internal network (in
 *    cloud it sits behind ACA internal ingress) and avoids any CORS setup.
 *
 * The route returns a small discriminated shape so the dashboard can render the
 * right state without guessing:
 *   { status: "disabled" }            -> PROMETHEUS_URL not set in this env
 *   { status: "success", data: ... }  -> passed through from Prometheus
 *   { status: "error", error: ... }   -> bad request, or Prometheus unreachable
 */
import { NextRequest, NextResponse } from "next/server";

// Trailing slashes are stripped so `${PROM}/api/...` never doubles up.
const PROM = (process.env.PROMETHEUS_URL || "").replace(/\/+$/, "");

// Backend session cookie name (api/auth/dependencies.py SESSION_COOKIE_NAME).
// The dashboard already lives behind the login gate; this is a matching, cheap
// presence check so the metrics proxy is not trivially open to anonymous
// callers. It is intentionally lightweight: the cookie is validated for real by
// the backend on every /stats call, and the backend /metrics it reads is itself
// only reachable on the internal network (or, in cloud, the same public ingress
// the backend already exposes).
const SESSION_COOKIE = "session_id";

/**
 * Proxy a Prometheus range query. Returns { status: "disabled" } when
 * PROMETHEUS_URL is unset, the upstream Prometheus JSON on success, or
 * { status: "error" } on a bad request or an unreachable Prometheus.
 */
export async function GET(req: NextRequest): Promise<NextResponse> {
  const authed =
    req.cookies.get(SESSION_COOKIE) != null || req.headers.get("x-api-key");
  if (!authed) {
    return NextResponse.json(
      { status: "error", error: "Not authenticated." },
      { status: 401 },
    );
  }

  // No Prometheus configured for this environment: report it plainly so the UI
  // shows a "not configured" note instead of a misleading network error.
  if (!PROM) {
    return NextResponse.json({ status: "disabled" });
  }

  const sp = req.nextUrl.searchParams;
  const query = sp.get("query");
  const start = sp.get("start");
  const end = sp.get("end");
  const step = sp.get("step");
  if (!query || !start || !end || !step) {
    return NextResponse.json(
      { status: "error", error: "query, start, end and step are required." },
      { status: 400 },
    );
  }

  const url =
    `${PROM}/api/v1/query_range?query=${encodeURIComponent(query)}` +
    `&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}` +
    `&step=${encodeURIComponent(step)}`;

  try {
    // 10s ceiling: a range query over a small TSDB is fast; if Prometheus is
    // wedged we want a clean error, not a hung dashboard tile.
    const upstream = await fetch(url, {
      signal: AbortSignal.timeout(10_000),
    });
    const body = await upstream.json().catch(() => null);
    if (!body) {
      return NextResponse.json(
        { status: "error", error: "Prometheus returned a non-JSON response." },
        { status: 502 },
      );
    }
    // Prometheus already uses {status, data|error}; forward it (and its status
    // code) unchanged so the client has one consistent shape to read.
    return NextResponse.json(body, { status: upstream.status });
  } catch {
    return NextResponse.json(
      { status: "error", error: "Could not reach Prometheus." },
      { status: 502 },
    );
  }
}
