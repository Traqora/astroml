import { useQuery } from '@tanstack/react-query'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { getTierComparison } from '../../api/loyalty'

export function TierComparisonChart() {
  const { data } = useQuery({ queryKey: ['tierComparison'], queryFn: getTierComparison })
  return (
    <div>
      <h2 style={{ margin: '8px 0' }}>Tier Comparison</h2>
      <div style={{ width: '100%', height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={data || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="tier" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="threshold" fill="#8884d8" />
            <Bar dataKey="multiplier" fill="#82ca9d" />
            <Bar dataKey="retention" fill="#ffc658" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
