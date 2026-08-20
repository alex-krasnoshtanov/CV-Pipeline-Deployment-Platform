import { describe, it, expect, vi, beforeEach } from "vitest";

/**
 * Unit tests for the Grad-CAM heatmap helper.
 *
 * viridis() and viridisGradientCss() are pure and tested directly. For
 * colorizeHeatmap(), Canvas and Image are mocked (jsdom has neither).
 */

describe("viridis colormap", () => {
  it("maps endpoints and midpoint to the anchor colours", async () => {
    const { viridis } = await import("@/lib/heatmap");
    expect(viridis(0)).toEqual([68, 1, 84]);
    expect(viridis(1)).toEqual([253, 231, 37]);
    expect(viridis(0.5)).toEqual([33, 145, 140]);
  });

  it("clamps values outside [0, 1]", async () => {
    const { viridis } = await import("@/lib/heatmap");
    expect(viridis(-5)).toEqual([68, 1, 84]);
    expect(viridis(5)).toEqual([253, 231, 37]);
  });
});

describe("viridisGradientCss", () => {
  it("builds a left-to-right gradient spanning all anchors", async () => {
    const { viridisGradientCss } = await import("@/lib/heatmap");
    const css = viridisGradientCss();
    expect(css.startsWith("linear-gradient(to right,")).toBe(true);
    expect(css).toContain("rgb(68, 1, 84) 0%");
    expect(css).toContain("rgb(253, 231, 37) 100%");
  });
});

describe("colorizeHeatmap", () => {
  const mockGetImageData = vi.fn();
  const mockPutImageData = vi.fn();
  const mockToDataURL = vi.fn();
  const mockContext = {
    drawImage: vi.fn(),
    getImageData: mockGetImageData,
    putImageData: mockPutImageData,
  };

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.resetModules();
    mockToDataURL.mockReturnValue("data:image/png;base64,RESULT");

    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      if (tag === "canvas") {
        return {
          width: 0,
          height: 0,
          getContext: () => mockContext,
          toDataURL: mockToDataURL,
        } as unknown as HTMLCanvasElement;
      }
      return document.createElement(tag);
    });

    // Image mock that fires onload as soon as src is assigned.
    vi.stubGlobal(
      "Image",
      class MockImage {
        width = 1;
        height = 1;
        crossOrigin = "";
        onload: (() => void) | null = null;
        onerror: (() => void) | null = null;
        private _src = "";
        get src() {
          return this._src;
        }
        set src(val: string) {
          this._src = val;
          setTimeout(() => this.onload?.(), 0);
        }
      },
    );
  });

  it("colours peak pixels with the hot map and a strong alpha", async () => {
    // Single channel grayscale=255 -> t=1 -> top of the hot map, high alpha.
    const heatPixels = new Uint8ClampedArray([255, 255, 255, 255]);
    mockGetImageData.mockReturnValueOnce({
      data: heatPixels,
      width: 1,
      height: 1,
    });

    const { colorizeHeatmap } = await import("@/lib/heatmap");
    const result = await colorizeHeatmap("AAAA");

    expect(result).toBe("data:image/png;base64,RESULT");
    expect(mockPutImageData).toHaveBeenCalled();
    // Hot map top anchor is [255, 245, 130]; alpha near alphaMax (0.95).
    expect([heatPixels[0], heatPixels[1], heatPixels[2]]).toEqual([255, 245, 130]);
    expect(heatPixels[3]).toBeGreaterThan(200);
  });

  it("makes weak attribution fully transparent", async () => {
    // grayscale=0 -> t=0, below the threshold -> alpha 0, colour untouched.
    const heatPixels = new Uint8ClampedArray([0, 0, 0, 255]);
    mockGetImageData.mockReturnValueOnce({
      data: heatPixels,
      width: 1,
      height: 1,
    });

    const { colorizeHeatmap } = await import("@/lib/heatmap");
    await colorizeHeatmap("AAAA");

    expect(heatPixels[3]).toBe(0);
    expect([heatPixels[0], heatPixels[1], heatPixels[2]]).toEqual([0, 0, 0]);
  });
});
