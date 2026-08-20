import { describe, it, expect, vi, beforeEach } from "vitest";

/**
 * Unit tests for the canvas overlay utility.
 *
 * Canvas and Image APIs are mocked since jsdom doesn't support them.
 * The tests verify the function correctly composes the overlay.
 */

const mockGetImageData = vi.fn();
const mockPutImageData = vi.fn();
const mockDrawImage = vi.fn();
const mockToDataURL = vi.fn().mockReturnValue("data:image/png;base64,RESULT");

const mockContext = {
  drawImage: mockDrawImage,
  getImageData: mockGetImageData,
  putImageData: mockPutImageData,
};

beforeEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();

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

  // Simple Image mock that triggers onload on src set
  vi.stubGlobal("Image", class MockImage {
    width = 100;
    height = 100;
    crossOrigin = "";
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    private _src = "";

    get src() { return this._src; }
    set src(val: string) {
      this._src = val;
      setTimeout(() => this.onload?.(), 0);
    }
  });
});

describe("overlayMask", () => {
  it("creates canvases and returns a data URL", async () => {
    const maskPixels = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]);
    const origPixels = new Uint8ClampedArray([100, 100, 100, 255, 100, 100, 100, 255]);

    mockGetImageData
      .mockReturnValueOnce({ data: origPixels, width: 2, height: 1 })
      .mockReturnValueOnce({ data: maskPixels, width: 2, height: 1 });

    const { overlayMask } = await import("@/lib/overlay");
    const result = await overlayMask("blob:test", "AAAA");

    expect(result).toBe("data:image/png;base64,RESULT");
    expect(mockToDataURL).toHaveBeenCalledWith("image/png");
    expect(mockPutImageData).toHaveBeenCalled();
  });

  it("accepts custom colour and alpha", async () => {
    const maskPixels = new Uint8ClampedArray([128, 128, 128, 255]);
    const origPixels = new Uint8ClampedArray([200, 200, 200, 255]);

    mockGetImageData
      .mockReturnValueOnce({ data: origPixels, width: 1, height: 1 })
      .mockReturnValueOnce({ data: maskPixels, width: 1, height: 1 });

    const { overlayMask } = await import("@/lib/overlay");
    const result = await overlayMask("blob:test", "AAAA", [0, 255, 0], 200);

    expect(result).toBe("data:image/png;base64,RESULT");
  });
});
