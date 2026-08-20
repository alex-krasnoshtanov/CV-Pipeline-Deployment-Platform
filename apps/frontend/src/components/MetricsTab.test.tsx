import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import MetricsTab from "@/components/MetricsTab";
import { type InferenceResult } from "@/lib/api";

/**
 * Unit tests for the Metrics tab.
 *
 * Pure presentational: render with a fixed InferenceResult and assert the
 * headline numbers, the alert-threshold labels, and the per-landmark rows.
 */

function makeResult(overrides: Partial<InferenceResult> = {}): InferenceResult {
  return {
    pipeline_version: "0.1.0",
    model_version: "unet-v1",
    timestamp: "2026-04-20T12:00:00Z",
    image_filename: "test.png",
    image_width_px: 256,
    image_height_px: 256,
    metadata: { plate_id: "PL-001", experiment_id: null, timestamp: null },
    mask_b64: "AAAA",
    mask_confidence: 0.91,
    landmark_count: 2,
    landmarks: [
      { id: 0, x: 10, y: 20, confidence: 0.92 },
      { id: 1, x: 30, y: 40, confidence: 0.4 },
    ],
    prediction_id: "p1",
    ...overrides,
  };
}

describe("MetricsTab", () => {
  it("shows the headline metrics", () => {
    render(<MetricsTab result={makeResult()} />);

    expect(screen.getByText("2")).toBeInTheDocument(); // landmark count
    expect(screen.getByText("0.910")).toBeInTheDocument(); // mask confidence
    expect(screen.getByText("unet-v1")).toBeInTheDocument();
    expect(screen.getByText("0.1.0")).toBeInTheDocument();
  });

  it("labels the confidence and renders the alert thresholds", () => {
    render(<MetricsTab result={makeResult({ mask_confidence: 0.91 })} />);

    expect(screen.getByText("High")).toBeInTheDocument();
    expect(screen.getByText("uncertain")).toBeInTheDocument();
    expect(screen.getByText("alert")).toBeInTheDocument();
  });

  it("renders one row per landmark with its confidence", () => {
    render(<MetricsTab result={makeResult()} />);

    expect(screen.getByText("Tip 0")).toBeInTheDocument();
    expect(screen.getByText("Tip 1")).toBeInTheDocument();
    expect(screen.getByText("0.920")).toBeInTheDocument();
    expect(screen.getByText("0.400")).toBeInTheDocument();
  });
});
