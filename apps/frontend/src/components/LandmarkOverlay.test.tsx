import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import LandmarkOverlay from "@/components/LandmarkOverlay";
import { type Landmark } from "@/lib/api";

/**
 * Unit tests for the landmark circle overlay.
 *
 * The component is pure SVG, so assertions count the rendered <circle> and
 * <text> nodes and check the viewBox maps to the image's pixel space.
 */

const landmarks: Landmark[] = [
  { id: 0, x: 10, y: 20, confidence: 0.92 },
  { id: 1, x: 30, y: 40, confidence: 0.45 },
];

describe("LandmarkOverlay", () => {
  it("draws a ring + dot and an id label per landmark", () => {
    const { container } = render(
      <LandmarkOverlay landmarks={landmarks} width={200} height={100} />,
    );
    // Two circles (ring + centre dot) per landmark.
    expect(container.querySelectorAll("circle")).toHaveLength(
      landmarks.length * 2,
    );
    expect(container.querySelectorAll("text")).toHaveLength(landmarks.length);

    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("viewBox")).toBe("0 0 200 100");
  });

  it("hides labels when showLabels is false but keeps the circles", () => {
    const { container } = render(
      <LandmarkOverlay
        landmarks={landmarks}
        width={200}
        height={100}
        showLabels={false}
      />,
    );
    expect(container.querySelectorAll("text")).toHaveLength(0);
    expect(container.querySelectorAll("circle")).toHaveLength(
      landmarks.length * 2,
    );
  });

  it("renders no markers for an empty landmark list", () => {
    const { container } = render(
      <LandmarkOverlay landmarks={[]} width={50} height={50} />,
    );
    expect(container.querySelectorAll("circle")).toHaveLength(0);
  });
});
