import React from 'react'
import { SCENARIOS } from '../data/scenarios'
import { IncidentReportCard } from '../components/IncidentReportCard'

export const IncidentReport: React.FC<{ onNavigate?: (page: string) => void }> = ({
  onNavigate,
}) => {
  const selectedScenario = SCENARIOS[0] // GitHub poisoned PR

  const handleCopy = () => {
    const summary = `Incident: ${selectedScenario.title}
Decision: ${selectedScenario.decision} (${selectedScenario.confidence}%)
Risk Evidence: ${selectedScenario.riskVectors.join('; ')}
Recommended: Require maintainer verification, inspect CI secret permissions, block merge until package provenance confirmed.`
    navigator.clipboard.writeText(summary).then(() => {
      alert('Summary copied to clipboard!')
    })
  }

  const handleExport = () => {
    const report = `ShadowOps Incident Report
========================================
Incident: ${selectedScenario.title}
Decision: ${selectedScenario.decision}
Confidence: ${selectedScenario.confidence}%

Risk Evidence:
${selectedScenario.riskVectors.map((v) => `- ${v}`).join('\n')}

Recommended Action:
Require maintainer verification, inspect CI secret permissions, and block merge until package provenance is confirmed.

MITRE Mapping:
Tactic: ${selectedScenario.mireTactic}
Technique: ${selectedScenario.mireTechnique}
`
    const blob = new Blob([report], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `incident-report-${selectedScenario.id}.txt`
    a.click()
  }

  const handleRunAnother = () => {
    onNavigate?.('mission-control')
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Incident Report</h2>
        <p className="text-slate-400 mt-1">Detailed analysis and recommendations</p>
      </div>

      <IncidentReportCard
        scenario={selectedScenario}
        onCopy={handleCopy}
        onExport={handleExport}
        onRun={handleRunAnother}
      />
    </div>
  )
}
