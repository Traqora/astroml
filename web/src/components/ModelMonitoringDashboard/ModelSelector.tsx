import { memo } from 'react'
import { useTranslation } from 'react-i18next'

interface ModelSelectorProps {
  selectedModel: string
  onModelChange: (model: string) => void
  models: string[]
}

export const ModelSelector = memo(function ModelSelector({
  selectedModel,
  onModelChange,
  models,
}: ModelSelectorProps) {
  const { t } = useTranslation()

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <label style={{ fontSize: 14, color: '#555' }}>
        {t('monitoring.model_selector.label')}:
      </label>
      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        style={{
          padding: '6px 12px',
          borderRadius: 6,
          border: '1px solid #d1d5db',
          fontSize: 14,
          background: '#fff',
          cursor: 'pointer',
        }}
      >
        {models.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
    </div>
  )
})