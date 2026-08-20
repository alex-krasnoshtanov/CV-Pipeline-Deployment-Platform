/**
 * Canvas-based segmentation mask overlay.
 *
 * Replaces the PIL-based overlay from the old Streamlit frontend.
 * Decodes a base64-encoded grayscale mask and blends a semi-transparent
 * coloured overlay onto the original image using an offscreen canvas.
 *
 * @param originalSrc - URL or blob URL of the original image.
 * @param maskB64 - Base64-encoded PNG mask (grayscale).
 * @param colour - RGB tuple for the overlay colour.
 * @param alpha - Opacity of the overlay (0–255).
 * @returns A data URL (`image/png`) of the composited result.
 */
export async function overlayMask(
  originalSrc: string,
  maskB64: string,
  colour: [number, number, number] = [255, 0, 0],
  alpha: number = 120
): Promise<string> {
  // Load original image.
  const original = await loadImage(originalSrc);
  const w = original.width;
  const h = original.height;

  // Load mask image.
  const maskUrl = `data:image/png;base64,${maskB64}`;
  const mask = await loadImage(maskUrl);

  // Draw original.
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(original, 0, 0, w, h);

  // Draw mask to temp canvas to read pixel data.
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = w;
  maskCanvas.height = h;
  const maskCtx = maskCanvas.getContext("2d")!;
  maskCtx.drawImage(mask, 0, 0, w, h);

  const maskData = maskCtx.getImageData(0, 0, w, h);
  const overlayData = ctx.getImageData(0, 0, w, h);
  const od = overlayData.data;
  const md = maskData.data;

  for (let i = 0; i < md.length; i += 4) {
    // Mask is grayscale; treat R channel as the mask intensity.
    const maskVal = md[i];
    if (maskVal > 0) {
      const a = (alpha * maskVal) / 255;
      const invA = 1 - a / 255;
      od[i] = Math.round(od[i] * invA + colour[0] * (a / 255));
      od[i + 1] = Math.round(od[i + 1] * invA + colour[1] * (a / 255));
      od[i + 2] = Math.round(od[i + 2] * invA + colour[2] * (a / 255));
    }
  }

  ctx.putImageData(overlayData, 0, 0);
  return canvas.toDataURL("image/png");
}

/**
 * Load an image from a URL and return the decoded HTMLImageElement.
 *
 * @param src - Image source URL (http, blob, or data URI).
 * @returns Resolved HTMLImageElement once the image has loaded.
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

