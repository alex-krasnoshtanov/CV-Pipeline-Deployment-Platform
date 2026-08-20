/**
 * Catch-all proxy route.
 *
 * Forwards every /api/* request to the FastAPI backend. This solves two
 * problems at once:
 *
 * 1. BACKEND_URL is a plain server-side env var — no NEXT_PUBLIC_ prefix —
 *    so it's read at runtime, not inlined at build time. The Docker image
 *    works in any environment without rebuilding.
 *
 * 2. The browser only talks to the Next.js origin, so session cookies are
 *    always same-origin. No CORS configuration needed.
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND = (
  process.env.BACKEND_URL || "http://localhost:8000"
).replace(/\/+$/, "");

/** Forward a request to the FastAPI backend and stream the response back. */
async function proxy(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
): Promise<NextResponse> {
  const { path } = await params;
  const target = `${BACKEND}/${path.join("/")}${req.nextUrl.search}`;

  // Forward only the headers the backend cares about.
  const headers = new Headers();
  for (const key of ["content-type", "cookie", "accept", "x-api-key"]) {
    const val = req.headers.get(key);
    if (val) headers.set(key, val);
  }

  let body: BodyInit | null = null;
  if (req.method !== "GET" && req.method !== "HEAD") {
    // .blob() handles both JSON and multipart/form-data correctly.
    body = await req.blob();
  }

  const upstream = await fetch(target, {
    method: req.method,
    headers,
    body,
  });

  // Stream the response back, forwarding status and relevant headers.
  const res = new NextResponse(upstream.body, { status: upstream.status });

  const ct = upstream.headers.get("content-type");
  if (ct) res.headers.set("content-type", ct);

  // set-cookie can appear multiple times; getSetCookie() returns all of them.
  for (const cookie of upstream.headers.getSetCookie?.() ?? []) {
    res.headers.append("set-cookie", cookie);
  }

  return res;
}

export {
  proxy as GET,
  proxy as POST,
  proxy as PUT,
  proxy as PATCH,
  proxy as DELETE,
};
