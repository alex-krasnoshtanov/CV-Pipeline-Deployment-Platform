/**
 * Grad-CAM heatmap colourisation.
 *
 * The backend returns the Seg-Grad-CAM heatmap as a single-channel grayscale
 * PNG (0 = no attribution, 255 = peak), deliberately *uncoloured* so the
 * frontend owns the colour map and opacity. We turn that grayscale map into a
 * standalone, transparent-background colour overlay (a PNG data URL). The
 * ExplainTab then stacks it over the original image as its own layer, so
 * opacity is a cheap CSS property on that layer instead of a per-pixel
 * recompute -- dragging the opacity control no longer re-runs this function.
 *
 * Visibility design (why the old overlay looked faint):
 *  - alpha used to scale linearly with intensity (alpha * t), so mid-attribution
 *    pixels were almost transparent. We instead use an alpha *floor*: any pixel
 *    above a small threshold gets a strong, clearly visible opacity.
 *  - a gamma lift (t ** gamma, gamma < 1) brightens the mid-range so regions,
 *    not just the single hottest pixel, are readable.
 *  - the default "hot" map (deep red -> orange -> bright yellow) reads on the
 *    near-black HADES plate; viridis' dark-purple low end did not. "warmer =
 *    more" still holds, matching the tab's caption.
 */

/** Named colour maps available for the heatmap overlay. */
export type ColormapName = "hot" | "viridis";

// Five viridis anchor stops; intermediate values are linearly interpolated.
const VIRIDIS: ReadonlyArray<readonly [number, number, number]> = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

// "hot": warm, bright, and legible on a dark background. Ends on bright yellow
// so the existing "warmer (yellow) areas contributed more" caption stays true.
const HOT: ReadonlyArray<readonly [number, number, number]> = [
  [60, 0, 0],
  [180, 30, 0],
  [240, 90, 0],
  [255, 170, 30],
  [255, 245, 130],
];

const MAPS: Record<
  ColormapName,
  ReadonlyArray<readonly [number, number, number]>
> = {
  hot: HOT,
  viridis: VIRIDIS,
};

/** Interpolate an anchor table at t in [0, 1] (clamped). */
function sample(
  anchors: ReadonlyArray<readonly [number, number, number]>,
  t: number,
): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (anchors.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = anchors[i];
  const b = anchors[Math.min(i + 1, anchors.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/**
 * Map a normalised value in [0, 1] to an RGB colour for the named map.
 *
 * @param name - Colormap to use.
 * @param t - Intensity in [0, 1]; values outside are clamped.
 * @returns An `[r, g, b]` tuple in 0-255.
 */
export function colormapColor(
  name: ColormapName,
  t: number,
): [number, number, number] {
  return sample(MAPS[name], t);
}

/**
 * Map a normalised value in [0, 1] to an RGB viridis colour.
 *
 * Kept as a named export because other code and tests import it directly.
 *
 * @param t - Intensity in [0, 1]; values outside are clamped.
 * @returns An `[r, g, b]` tuple in 0-255.
 */
export function viridis(t: number): [number, number, number] {
  return sample(VIRIDIS, t);
}

/** CSS gradient string for a legend bar of the named map. */
export function gradientCss(name: ColormapName = "hot"): string {
  const anchors = MAPS[name];
  const stops = anchors.map(
    ([r, g, b], idx) =>
      `rgb(${r}, ${g}, ${b}) ${(idx / (anchors.length - 1)) * 100}%`,
  );
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

/** CSS gradient string for a viridis legend bar (kept for existing callers). */
export function viridisGradientCss(): string {
  return gradientCss("viridis");
}

/** Options for {@link colorizeHeatmap}. All have sensible defaults. */
export interface ColorizeOptions {
  /** Colormap to apply. Default "hot". */
  colormap?: ColormapName;
  /** Gamma applied to intensity before colouring (<1 brightens mid-tones). */
  gamma?: number;
  /** Intensity at or below which a pixel is fully transparent. */
  threshold?: number;
  /** Minimum opacity for any pixel above the threshold. */
  alphaFloor?: number;
  /** Maximum opacity, reached at peak intensity. */
  alphaMax?: number;
}

/**
 * Turn a grayscale Grad-CAM heatmap into a standalone colour overlay.
 *
 * Computed once per explanation; the result is a transparent-background PNG
 * the caller stacks over the original image. Opacity is then a CSS property on
 * that layer, so this expensive pixel pass does not re-run on every slider move.
 *
 * @param heatmapB64 - Base64 grayscale PNG heatmap (same size as the original).
 * @param opts - Colour/opacity options; see {@link ColorizeOptions}.
 * @returns A data URL (`image/png`) of the transparent colour overlay.
 */
export async function colorizeHeatmap(
  heatmapB64: string,
  opts: ColorizeOptions = {},
): Promise<string> {
  const {
    colormap = "hot",
    gamma = 0.6,
    threshold = 0.08,
    alphaFloor = 0.45,
    alphaMax = 0.95,
  } = opts;

  const heatmap = await loadImage(`data:image/png;base64,${heatmapB64}`);
  const w = heatmap.width;
  const h = heatmap.height;

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(heatmap, 0, 0, w, h);

  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  const anchors = MAPS[colormap];

  for (let i = 0; i < d.length; i += 4) {
    const t = d[i] / 255; // grayscale: R channel carries the intensity
    if (t <= threshold) {
      d[i + 3] = 0; // weak attribution -> fully transparent
      continue;
    }
    const tc = Math.pow(t, gamma);
    const [r, g, b] = sample(anchors, tc);
    d[i] = r;
    d[i + 1] = g;
    d[i + 2] = b;
    d[i + 3] = Math.round(255 * (alphaFloor + (alphaMax - alphaFloor) * tc));
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

/**
 * Load an image from a URL and resolve once decoded.
 *
 * @param src - Image source URL (http, blob, or data URI).
 * @returns The decoded HTMLImageElement.
 */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
