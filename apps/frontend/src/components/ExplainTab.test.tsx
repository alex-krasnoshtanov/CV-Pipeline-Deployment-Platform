import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { type ExplanationResult } from "@/lib/api";

/**
 * Unit tests for the Explain tab.
 *
 * postExplain (network) and colorizeHeatmap (canvas) are mocked, so the tests
 * focus on the component's behaviour: button gating, the fetch-then-render
 * flow, and error handling.
 */

const { mockPostExplain, mockColorizeHeatmap } = vi.hoisted(() => ({
  mockPostExplain: vi.fn(),
  mockColorizeHeatmap: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  postExplain: mockPostExplain,
}));

vi.mock("@/lib/heatmap", () => ({
  colorizeHeatmap: mockColorizeHeatmap,
  gradientCss: () => "linear-gradient(to right, rgb(60, 0, 0) 0%)",
}));

import ExplainTab from "@/components/ExplainTab";

const explanation: ExplanationResult = {
  pipeline_version: "0.1.0",
  model_version: "unet-v1",
  timestamp: "2026-04-20T12:00:00Z",
  image_filename: "test.png",
  image_width_px: 256,
  image_height_px: 256,
  metadata: { plate_id: null, experiment_id: null, timestamp: null },
  method: "seg-grad-cam",
  target_layer: "decoder.blocks[-1]",
  downscaled: false,
  heatmap_peak: 0.42,
  heatmap_b64: "AAAA",
};

beforeEach(() => {
  vi.clearAllMocks();
  mockColorizeHeatmap.mockResolvedValue("data:image/png;base64,OVERLAY");
});

function makeFile(): File {
  return new File([new Uint8Array([1, 2, 3])], "test.png", {
    type: "image/png",
  });
}

describe("ExplainTab", () => {
  it("disables the button when no file is selected", () => {
    render(<ExplainTab file={null} previewUrl={null} />);
    expect(
      screen.getByRole("button", { name: /generate explanation/i }),
    ).toBeDisabled();
  });

  it("requests an explanation and renders the overlay + metadata", async () => {
    mockPostExplain.mockResolvedValue({
      data: explanation,
      error: null,
      status: 200,
    });
    const user = userEvent.setup();

    render(<ExplainTab file={makeFile()} previewUrl="blob:orig" />);
    await user.click(
      screen.getByRole("button", { name: /generate explanation/i }),
    );

    await waitFor(() => expect(mockPostExplain).toHaveBeenCalledTimes(1));

    expect(await screen.findByText("seg-grad-cam")).toBeInTheDocument();
    expect(screen.getByText("decoder.blocks[-1]")).toBeInTheDocument();

    await waitFor(() =>
      expect(
        screen.getByAltText("Grad-CAM heatmap overlay"),
      ).toBeInTheDocument(),
    );
    expect(mockColorizeHeatmap).toHaveBeenCalledWith("AAAA");
  });

  it("shows an error when the backend call fails", async () => {
    mockPostExplain.mockResolvedValue({
      data: null,
      error: "Explanation failed.",
      status: 500,
    });
    const user = userEvent.setup();

    render(<ExplainTab file={makeFile()} previewUrl="blob:orig" />);
    await user.click(
      screen.getByRole("button", { name: /generate explanation/i }),
    );

    expect(await screen.findByText("Explanation failed.")).toBeInTheDocument();
  });

  it("toggles the overlay layer without re-colourising", async () => {
    mockPostExplain.mockResolvedValue({
      data: explanation,
      error: null,
      status: 200,
    });
    const user = userEvent.setup();

    render(<ExplainTab file={makeFile()} previewUrl="blob:orig" />);
    await user.click(
      screen.getByRole("button", { name: /generate explanation/i }),
    );
    await screen.findByAltText("Grad-CAM heatmap overlay");

    // colourise runs exactly once for the fetched explanation.
    expect(mockColorizeHeatmap).toHaveBeenCalledTimes(1);

    // Hiding then showing the overlay must not trigger another colourise.
    await user.click(screen.getByRole("button", { name: /hide overlay/i }));
    await user.click(screen.getByRole("button", { name: /show overlay/i }));
    expect(mockColorizeHeatmap).toHaveBeenCalledTimes(1);
  });
});
