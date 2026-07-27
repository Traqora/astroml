import { useEffect, useRef, useState, useCallback, memo } from 'react'
import * as d3 from 'd3'
import { useTransactionUpdates } from '../../hooks/useWebSocket'

interface GraphNode extends d3.SimulationNodeDatum {
  id: string
  label: string
  type: 'source' | 'destination'
  balance: number
}

interface GraphEdge {
  source: string
  target: string
  amount: number
  asset: string
  timestamp: string
}

interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

const ASSET_COLORS: Record<string, string> = {
  XLM: '#3b82f6',
  USDC: '#22c55e',
  ETH: '#8b5cf6',
  BTC: '#f59e0b',
}

function getAssetColor(asset: string): string {
  return ASSET_COLORS[asset] || '#6b7280'
}

const MAX_NODES = 80
const MAX_EDGES = 200

export const GraphVisualization = memo(function GraphVisualization() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] })
  const [searchTerm, setSearchTerm] = useState('')
  const [layout, setLayout] = useState<'force' | 'circular'>('force')

  const addTransaction = useCallback((tx: any) => {
    setGraphData((prev) => {
      const sourceId = tx.sourceAccount || `src-${Math.random().toString(36).slice(2, 8)}`
      const targetId = tx.destinationAccount || `dst-${Math.random().toString(36).slice(2, 8)}`

      const newNodes: GraphNode[] = [...prev.nodes]
      const newEdges: GraphEdge[] = [...prev.edges]

      if (!newNodes.find((n) => n.id === sourceId)) {
        newNodes.push({ id: sourceId, label: sourceId.slice(0, 8), type: 'source', balance: tx.amount || 0 })
      }
      if (!newNodes.find((n) => n.id === targetId)) {
        newNodes.push({ id: targetId, label: targetId.slice(0, 8), type: 'destination', balance: 0 })
      }

      newEdges.push({
        source: sourceId,
        target: targetId,
        amount: tx.amount || 0,
        asset: tx.assetCode || 'XLM',
        timestamp: tx.timestamp || new Date().toISOString(),
      })

      if (newNodes.length > MAX_NODES) newNodes.splice(0, newNodes.length - MAX_NODES)
      if (newEdges.length > MAX_EDGES) newEdges.splice(0, newEdges.length - MAX_EDGES)

      return { nodes: newNodes, edges: newEdges }
    })
  }, [])

  useTransactionUpdates(addTransaction)

  const filteredData = useCallback(() => {
    if (!searchTerm) return graphData
    const term = searchTerm.toLowerCase()
    const matchingIds = new Set(
      graphData.nodes.filter((n) => n.label.toLowerCase().includes(term)).map((n) => n.id)
    )
    return {
      nodes: graphData.nodes.filter((n) => matchingIds.has(n.id)),
      edges: graphData.edges.filter((e) => matchingIds.has(e.source) && matchingIds.has(e.target)),
    }
  }, [graphData, searchTerm])

  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    const width = containerRef.current?.clientWidth || 800
    const height = 500

    svg.selectAll('*').remove()

    const data = filteredData()
    if (data.nodes.length === 0) return

    const g = svg.append('g')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    const simulation = d3.forceSimulation<GraphNode>(data.nodes)
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))

    if (layout === 'force') {
      simulation
        .force('link', d3.forceLink<GraphNode, GraphEdge>(data.edges).id((d) => d.id).distance(100))
        .force('collision', d3.forceCollide().radius(20))
    } else {
      const cx = width / 2
      const cy = height / 2
      const radius = Math.min(width, height) / 2 - 40
      data.nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / data.nodes.length
        node.x = cx + radius * Math.cos(angle)
        node.y = cy + radius * Math.sin(angle)
      })
      simulation.stop()
    }

    const link = g.append('g')
      .selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('stroke', (d) => getAssetColor(d.asset))
      .attr('stroke-width', (d) => Math.max(1, Math.min(d.amount / 1000, 6)))
      .attr('stroke-opacity', 0.6)

    const node = g.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', 8)
      .attr('fill', (d) => d.type === 'source' ? '#3b82f6' : '#22c55e')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .call((d3.drag<SVGCircleElement, GraphNode>() as any)
          .on('start', (event: any, d: any) => {
            if (!event.active) simulation.alphaTarget(0.3).restart()
            d.fx = d.x
            d.fy = d.y
          })
          .on('drag', (event: any, d: any) => {
            d.fx = event.x
            d.fy = event.y
          })
          .on('end', (event: any, d: any) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = null
            d.fy = null
          })
      )

    node.append('title')
      .text((d) => `${d.id}\nBalance: ${d.balance}`)

    const label = g.append('g')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .text((d) => d.label)
      .attr('font-size', 10)
      .attr('dx', 10)
      .attr('dy', 4)
      .attr('fill', 'var(--text-primary, #1a202c)')

    if (layout === 'force') {
      simulation.on('tick', () => {
        link
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y)

        node
          .attr('cx', (d: any) => d.x)
          .attr('cy', (d: any) => d.y)

        label
          .attr('x', (d: any) => d.x)
          .attr('y', (d: any) => d.y)
      })
    }

    return () => {
      simulation.stop()
    }
  }, [graphData, searchTerm, layout, filteredData])

  const exportImage = useCallback(() => {
    if (!svgRef.current) return
    const svgData = new XMLSerializer().serializeToString(svgRef.current)
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const img = new Image()
    const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)

    img.onload = () => {
      canvas.width = img.width * 2
      canvas.height = img.height * 2
      ctx!.scale(2, 2)
      ctx!.fillStyle = '#ffffff'
      ctx!.fillRect(0, 0, canvas.width, canvas.height)
      ctx!.drawImage(img, 0, 0)
      URL.revokeObjectURL(url)
      const a = document.createElement('a')
      a.download = 'graph-visualization.png'
      a.href = canvas.toDataURL('image/png')
      a.click()
    }
    img.src = url
  }, [])

  return (
    <div>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        gap: 12,
        flexWrap: 'wrap',
        marginBottom: 12,
      }}>
        <h2 style={{ margin: 0 }}>Transaction Network Graph</h2>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              padding: '6px 10px',
              borderRadius: 4,
              border: '1px solid var(--border-color, #ddd)',
              background: 'var(--bg-primary, #fff)',
              color: 'var(--text-primary, #1a202c)',
              fontSize: 13,
              width: 160,
            }}
          />
          <select
            value={layout}
            onChange={(e) => setLayout(e.target.value as 'force' | 'circular')}
            style={{
              padding: '6px 10px',
              borderRadius: 4,
              border: '1px solid var(--border-color, #ddd)',
              background: 'var(--bg-primary, #fff)',
              color: 'var(--text-primary, #1a202c)',
              fontSize: 13,
            }}
          >
            <option value="force">Force-Directed</option>
            <option value="circular">Circular</option>
          </select>
          <button
            onClick={exportImage}
            style={{
              padding: '6px 12px',
              borderRadius: 4,
              border: '1px solid var(--border-color, #ddd)',
              background: 'var(--bg-card, #fff)',
              color: 'var(--text-primary, #1a202c)',
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 600,
            }}
          >
            Export PNG
          </button>
        </div>
      </div>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: 500,
          border: '1px solid var(--border-color, #e2e8f0)',
          borderRadius: 8,
          overflow: 'hidden',
          background: 'var(--bg-card, #fff)',
        }}
      >
        <svg ref={svgRef} width="100%" height="100%" style={{ display: 'block' }} />
      </div>
    </div>
  )
})
