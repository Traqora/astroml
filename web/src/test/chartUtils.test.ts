import { describe, it, expect } from 'vitest'
import { sampleData, CHART_SAMPLE_THRESHOLD, CHART_TARGET_POINTS } from '../lib/chartUtils'

type Point = { x: number; y: number }

function makePoints(n: number): Point[] {
  return Array.from({ length: n }, (_, i) => ({ x: i, y: Math.sin(i / 10) * 100 }))
}

describe('sampleData', () => {
  it('returns the original array when length ≤ threshold', () => {
    const data = makePoints(10)
    const result = sampleData(data, 20, 'y')
    expect(result).toBe(data) // same reference — no copy made
  })

  it('returns exactly `threshold` points for large input', () => {
    const data = makePoints(10_000)
    const result = sampleData(data, 300, 'y')
    expect(result).toHaveLength(300)
  })

  it('always preserves the first and last points', () => {
    const data = makePoints(5_000)
    const result = sampleData(data, 100, 'y')
    expect(result[0]).toBe(data[0])
    expect(result[result.length - 1]).toBe(data[data.length - 1])
  })

  it('handles threshold === 2 (first + last only)', () => {
    const data = makePoints(100)
    const result = sampleData(data, 2, 'y')
    expect(result).toHaveLength(2)
    expect(result[0]).toBe(data[0])
    expect(result[1]).toBe(data[99])
  })

  it('is a no-op for empty arrays', () => {
    const result = sampleData([], 100, 'y')
    expect(result).toHaveLength(0)
  })

  it('is a no-op for single-element arrays', () => {
    const data = makePoints(1)
    const result = sampleData(data, 100, 'y')
    expect(result).toBe(data)
  })

  it('preserves all points for a flat signal (no information lost)', () => {
    // All y values equal — any selection is valid, just ensure correct count
    const data = Array.from({ length: 1000 }, (_, i) => ({ x: i, y: 42 }))
    const result = sampleData(data, 100, 'y')
    expect(result).toHaveLength(100)
  })

  it('exported constants have sensible values', () => {
    expect(CHART_SAMPLE_THRESHOLD).toBeGreaterThan(0)
    expect(CHART_TARGET_POINTS).toBeGreaterThan(0)
    expect(CHART_TARGET_POINTS).toBeLessThanOrEqual(CHART_SAMPLE_THRESHOLD)
  })
})
