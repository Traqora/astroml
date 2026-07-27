/**
 * Chart performance utilities for issue #291.
 *
 * Provides:
 *  - sampleData       – reduce large datasets to a target point count (LTTB algorithm)
 *  - createChartConfig – returns Recharts props that maximise canvas-layer performance
 */

// ---------------------------------------------------------------------------
// Largest-Triangle-Three-Buckets (LTTB) downsampling
// Retains the visual shape of a series while drastically reducing point count.
// Reference: Sveinn Steinarsson, "Downsampling Time Series for Visual Representation" (2013)
// ---------------------------------------------------------------------------

export interface DataPoint {
  [key: string]: number | string
}

/**
 * Downsample an array of data points to at most `threshold` points using LTTB.
 * If the array is already ≤ threshold, it is returned unchanged.
 *
 * @param data      - source data array
 * @param threshold - maximum number of output points (must be ≥ 2)
 * @param yKey      - name of the numeric y-axis field used to compute triangles
 */
export function sampleData<T extends DataPoint>(data: T[], threshold: number, yKey: keyof T): T[] {
  if (threshold < 2 || data.length <= threshold) return data

  const sampled: T[] = []
  // Always include first and last points
  sampled.push(data[0])

  const bucketSize = (data.length - 2) / (threshold - 2)

  let a = 0 // index of the previously selected point

  for (let i = 0; i < threshold - 2; i++) {
    // Calculate bucket range
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1
    const bucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length - 1)

    // Average point for the next bucket (used as the third triangle vertex)
    const nextBucketStart = bucketEnd
    const nextBucketEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, data.length - 1)
    let avgX = 0
    let avgY = 0
    const nextBucketSize = nextBucketEnd - nextBucketStart
    if (nextBucketSize > 0) {
      for (let j = nextBucketStart; j < nextBucketEnd; j++) {
        avgX += j
        avgY += Number(data[j][yKey])
      }
      avgX /= nextBucketSize
      avgY /= nextBucketSize
    } else {
      avgX = nextBucketStart
      avgY = Number(data[nextBucketStart]?.[yKey] ?? 0)
    }

    // Point A (previously selected)
    const ax = a
    const ay = Number(data[a][yKey])

    // Select the point in the current bucket with the largest triangle area
    let maxArea = -1
    let selectedIndex = bucketStart

    for (let j = bucketStart; j < bucketEnd; j++) {
      const bx = j
      const by = Number(data[j][yKey])
      // Triangle area (×2, sign doesn't matter)
      const area = Math.abs((ax - avgX) * (by - ay) - (ax - bx) * (avgY - ay))
      if (area > maxArea) {
        maxArea = area
        selectedIndex = j
      }
    }

    sampled.push(data[selectedIndex])
    a = selectedIndex
  }

  sampled.push(data[data.length - 1])
  return sampled
}

// ---------------------------------------------------------------------------
// Recharts performance config
// ---------------------------------------------------------------------------

/**
 * Returns a set of Recharts-compatible props that squeeze out maximum rendering
 * performance.  Spread these onto the top-level chart component.
 *
 * - `isAnimationActive: false` avoids expensive CSS transitions on every render.
 * - explicit `width`/`height` (when inside a fixed container) lets the browser
 *   skip an extra layout pass that ResponsiveContainer would otherwise trigger.
 */
export function createChartConfig(options: { animate?: boolean } = {}) {
  return {
    isAnimationActive: options.animate ?? false,
    // Recharts uses SVG internally; these margin hints minimise
    // reflow by giving the SVG a stable bounding box
    margin: { top: 8, right: 16, left: 0, bottom: 0 },
  } as const
}

// ---------------------------------------------------------------------------
// Threshold constants
// ---------------------------------------------------------------------------

/** Point count above which sampleData should be applied. */
export const CHART_SAMPLE_THRESHOLD = 500

/** Default target point count after sampling. */
export const CHART_TARGET_POINTS = 300
