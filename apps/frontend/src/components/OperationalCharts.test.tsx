import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import OperationalCharts from "./OperationalCharts";

// recharts is mocked so the chart primitives render nothing in jsdom (they need
// real layout). The titles, range buttons and state notes under test are plain
// DOM rendered by OperationalCharts itself, not by recharts, so this is enough.
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
}));

vi.mock("@/lib/api", () => ({
  getMetricsRange: vi.fn(),
  postMonitoringCheck: vi.fn(),
}));

import {
  getMetricsRange,
  postMonitoringCheck,
  type PromRangeResponse,
} from "@/lib/api";

// A minimal single-series range result, the shape the proxy returns.
const matrix: PromRangeResponse = {
  status: "success",
  data: {
    resultType: "matrix",
    result: [
      {
        metric: {},
        values: [
          [1000, "1"],
          [1015, "2"],
        ],
      },
    ],
  },
};

describe("OperationalCharts", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows a not-configured note when Prometheus is disabled", async () => {
    // The proxy returns { status: "disabled" } when PROMETHEUS_URL is unset.
    vi.mocked(getMetricsRange).mockResolvedValue({ status: "disabled" });
    render(<OperationalCharts />);
    expect(await screen.findByText(/not configured/i)).toBeInTheDocument();
  });

  it("renders chart titles and range controls on success", async () => {
    vi.mocked(getMetricsRange).mockResolvedValue(matrix);
    render(<OperationalCharts />);

    expect(await screen.findByText("Request Rate (req/s)")).toBeInTheDocument();
    expect(screen.getByText("Error Rate (%)")).toBeInTheDocument();
    expect(screen.getByText("Latency (ms)")).toBeInTheDocument();
    expect(
      screen.getByText("Low-confidence Fraction"),
    ).toBeInTheDocument();

    expect(screen.getByRole("button", { name: "15m" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "1h" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "6h" })).toBeInTheDocument();
  });

  it("runs a drift check when the admin button is clicked", async () => {
    vi.mocked(getMetricsRange).mockResolvedValue(matrix);
    vi.mocked(postMonitoringCheck).mockResolvedValue({
      data: { drift: { alert: false } },
      error: null,
      status: 200,
    });
    render(<OperationalCharts canRunDriftCheck />);

    const btn = await screen.findByRole("button", {
      name: /run drift check/i,
    });
    fireEvent.click(btn);
    await waitFor(() => expect(postMonitoringCheck).toHaveBeenCalled());
  });

  it("hides the drift-check button without admin rights", async () => {
    vi.mocked(getMetricsRange).mockResolvedValue(matrix);
    render(<OperationalCharts />);
    await screen.findByText("Request Rate (req/s)");
    expect(
      screen.queryByRole("button", { name: /run drift check/i }),
    ).toBeNull();
  });
});
