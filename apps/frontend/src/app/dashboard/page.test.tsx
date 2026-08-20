import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { StatsData } from "@/lib/api";

/**
 * Tests for the monitoring dashboard page.
 *
 * recharts' ResponsiveContainer renders nothing at jsdom's 0 width, so
 * the entire recharts module is mocked to plain divs. This lets us assert
 * on headings, KPI values, and empty-state copy without SVG rendering issues.
 *
 * The auth check and getStats fetch are both mocked per test.
 */

// ── Mock recharts ────────────────────────────────────────────────────────────
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="chart">{children}</div>
  ),
  BarChart: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  Bar: () => null,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
}));

// ── Mock Next.js navigation ──────────────────────────────────────────────────
const mockReplace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace }),
}));
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

// ── Mock API module ──────────────────────────────────────────────────────────
vi.mock("@/lib/api", () => ({
  getStats: vi.fn(),
  getMe: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
  register: vi.fn(),
  getHealth: vi.fn(),
  postInfer: vi.fn(),
  postFeedback: vi.fn(),
  // OperationalCharts (rendered inside the page) calls this on mount. Resolve to
  // "disabled" so it renders its not-configured note instead of hitting the
  // network in jsdom; the operational KPIs under test come from getStats.
  getMetricsRange: vi.fn().mockResolvedValue({ status: "disabled" }),
}));

// ── Mock AuthContext ─────────────────────────────────────────────────────────
vi.mock("@/context/AuthContext", () => ({
  useAuth: vi.fn(),
}));

import { useAuth } from "@/context/AuthContext";
import { getStats } from "@/lib/api";
import DashboardPage from "./page";

const mockUser = {
  id: "user-1",
  name: "Test User",
  role: "researcher",
  email: "test@example.com",
};

function mockAuth(user: typeof mockUser | null, loading = false) {
  (useAuth as ReturnType<typeof vi.fn>).mockReturnValue({
    user,
    loading,
    logout: vi.fn(),
    login: vi.fn(),
    register: vi.fn(),
  });
}

function makeStats(overrides: Partial<StatsData> = {}): StatsData {
  return {
    business: {
      total_predictions: 42,
      avg_confidence: 0.78,
      confidence_histogram: [
        { bucket_label: "0.7–0.8", count: 15 },
        { bucket_label: "0.8–0.9", count: 27 },
      ],
      predictions_over_time: [
        { date: "2026-05-01", count: 5 },
        { date: "2026-05-02", count: 8 },
      ],
      landmark_distribution: [{ landmark_count: 5, count: 42 }],
      feedback: {
        good: 30,
        bad: 8,
        uncertain: 4,
        total: 42,
        bad_rate: 8 / 42,
      },
      per_model_version: [
        {
          model_version: "unet-v1",
          prediction_count: 42,
          avg_confidence: 0.78,
        },
      ],
    },
    operational: {
      serving_model_version: "unet-v1",
      prometheus_metrics_available: true,
      request_metrics: {
        total_requests: 120,
        error_rate: 0.05,
        avg_latency_ms: 42.5,
      },
    },
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("DashboardPage — unauthenticated", () => {
  it("redirects to /login when auth loading is done and user is null", async () => {
    mockAuth(null, false);
    (getStats as ReturnType<typeof vi.fn>).mockResolvedValue({
      data: null,
      error: null,
      status: 401,
    });
    render(<DashboardPage />);
    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith("/login");
    });
  });

  it("shows spinner while auth is loading", () => {
    mockAuth(null, true);
    render(<DashboardPage />);
    expect(document.querySelector(".spinner")).toBeTruthy();
  });
});

describe("DashboardPage — authenticated with data", () => {
  beforeEach(() => {
    mockAuth(mockUser);
    (getStats as ReturnType<typeof vi.fn>).mockResolvedValue({
      data: makeStats(),
      error: null,
      status: 200,
    });
  });

  it("renders the page title", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText("Monitoring Dashboard")
    ).toBeInTheDocument();
  });

  it("renders the Business Insights section heading", async () => {
    render(<DashboardPage />);
    expect(await screen.findByText("Business Insights")).toBeInTheDocument();
  });

  it("renders the Operational Monitoring section heading", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText("Operational Monitoring")
    ).toBeInTheDocument();
  });

  it("shows total predictions KPI", async () => {
    render(<DashboardPage />);
    // Value appears in both the KPI card and the model-version table.
    const matches = await screen.findAllByText("42");
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it("shows avg confidence KPI", async () => {
    render(<DashboardPage />);
    // Value appears in both the KPI card and the model-version table.
    const matches = await screen.findAllByText("0.780");
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it("shows serving model version", async () => {
    render(<DashboardPage />);
    // "unet-v1" appears in both the KPI card and the model-version table.
    const matches = await screen.findAllByText("unet-v1");
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it("shows prometheus available badge", async () => {
    render(<DashboardPage />);
    expect(await screen.findByText("✓ Available")).toBeInTheDocument();
  });

  it("renders landmark distribution chart title", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText("Landmark Count Distribution")
    ).toBeInTheDocument();
  });

  it("renders total requests KPI", async () => {
    render(<DashboardPage />);
    expect(await screen.findByText("120")).toBeInTheDocument();
  });

  it("renders error rate KPI", async () => {
    render(<DashboardPage />);
    expect(await screen.findByText("5.0%")).toBeInTheDocument();
  });

  it("renders avg latency KPI", async () => {
    render(<DashboardPage />);
    expect(await screen.findByText("42.5")).toBeInTheDocument();
  });

  it("renders nav link to /dashboard", async () => {
    render(<DashboardPage />);
    await screen.findByText("Monitoring Dashboard");
    const link = document.querySelector('a[href="/dashboard"]');
    expect(link).toBeTruthy();
  });

  it("renders nav link back to /", async () => {
    render(<DashboardPage />);
    await screen.findByText("Monitoring Dashboard");
    const link = document.querySelector('a[href="/"]');
    expect(link).toBeTruthy();
  });
});

describe("DashboardPage — empty database", () => {
  beforeEach(() => {
    mockAuth(mockUser);
    const emptyStats = makeStats();
    emptyStats.business.total_predictions = 0;
    emptyStats.business.avg_confidence = null;
    emptyStats.business.predictions_over_time = [];
    emptyStats.business.landmark_distribution = [];
    emptyStats.business.feedback = {
      good: 0,
      bad: 0,
      uncertain: 0,
      total: 0,
      bad_rate: 0,
    };
    emptyStats.operational.serving_model_version = null;
    emptyStats.operational.request_metrics = {
      total_requests: 0,
      error_rate: 0.0,
      avg_latency_ms: null,
    };
    (getStats as ReturnType<typeof vi.fn>).mockResolvedValue({
      data: emptyStats,
      error: null,
      status: 200,
    });
  });

  it("shows empty state for prediction volume chart", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText("No predictions in the last 30 days.")
    ).toBeInTheDocument();
  });

  it("shows empty state for landmark distribution chart", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText("No landmark data yet.")
    ).toBeInTheDocument();
  });

  it("shows empty state for feedback", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText("No feedback submitted yet.")
    ).toBeInTheDocument();
  });

  it("shows em dash for null avg confidence", async () => {
    render(<DashboardPage />);
    await screen.findByText("Monitoring Dashboard");
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThan(0);
  });
});

describe("DashboardPage — stats fetch error", () => {
  it("shows error message when fetch fails", async () => {
    mockAuth(mockUser);
    (getStats as ReturnType<typeof vi.fn>).mockResolvedValue({
      data: null,
      error: "Cannot reach the backend. Is it running?",
      status: 0,
    });
    render(<DashboardPage />);
    expect(
      await screen.findByText("Cannot reach the backend. Is it running?")
    ).toBeInTheDocument();
  });
});
