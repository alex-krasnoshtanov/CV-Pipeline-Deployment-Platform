"use client";

import { type Landmark } from "@/lib/api";
import { confidenceColor } from "@/lib/confidence";

/**
 * SVG overlay that draws a ring + centre dot + id label on each detected root
 * tip, colour-coded by confidence.
 *
 * The SVG ``viewBox`` is the image's natural pixel space, so landmark
 * coordinates (which the API returns in original pixel space) are used
 * directly as ``cx``/``cy`` with no manual scaling — the browser scales the
 * whole SVG to match the displayed image. The parent must position this inside
 * a ``position: relative`` container that also holds the <img>.
 */
export default function LandmarkOverlay({
  landmarks,
  width,
  height,
  showLabels = true,
}: {
  landmarks: Landmark[];
  width: number;
  height: number;
  showLabels?: boolean;
}) {
  // Scale ring size to the image so it looks consistent regardless of
  // resolution. These factors are tuned for HADES plate proportions.
  const longest = Math.max(width, height);
  const radius = Math.max(6, longest * 0.012);
  const strokeWidth = Math.max(1.5, longest * 0.003);
  const fontSize = Math.max(10, longest * 0.02);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
      aria-label="Detected root tip landmarks"
    >
      {landmarks.map((lm) => {
        const colour = confidenceColor(lm.confidence);
        return (
          <g key={lm.id}>
            <title>{`Tip ${lm.id} - confidence ${lm.confidence.toFixed(3)}`}</title>
            <circle
              cx={lm.x}
              cy={lm.y}
              r={radius}
              style={{ fill: "none", stroke: colour, strokeWidth }}
            />
            <circle
              cx={lm.x}
              cy={lm.y}
              r={strokeWidth}
              style={{ fill: colour }}
            />
            {showLabels && (
              <text
                x={lm.x + radius + strokeWidth}
                y={lm.y - radius}
                style={{ fill: colour, fontSize, fontWeight: 700 }}
              >
                {lm.id}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
