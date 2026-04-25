import React from 'react'
import { Check, X, Clock, GitFork } from 'lucide-react'

interface DecisionBadgeProps {
  decision: 'ALLOW' | 'BLOCK' | 'QUARANTINE' | 'FORK'
  confidence?: number
}

export const DecisionBadge: React.FC<DecisionBadgeProps> = ({
  decision,
  confidence,
}) => {
  const styles = {
    ALLOW: 'bg-green-900 text-green-300 border-green-700',
    BLOCK: 'bg-red-900 text-red-300 border-red-700',
    QUARANTINE: 'bg-amber-900 text-amber-300 border-amber-700',
    FORK: 'bg-purple-900 text-purple-300 border-purple-700',
  }

  const icons = {
    ALLOW: <Check className="w-4 h-4" />,
    BLOCK: <X className="w-4 h-4" />,
    QUARANTINE: <Clock className="w-4 h-4" />,
    FORK: <GitFork className="w-4 h-4" />,
  }

  return (
    <div className={`badge ${styles[decision]}`}>
      {icons[decision]}
      <span className="ml-2">{decision}</span>
      {confidence && <span className="ml-1 text-xs opacity-75">{confidence}%</span>}
    </div>
  )
}

export const MetricCard: React.FC<{
  label: string
  value: string | number
  icon?: React.ReactNode
  variant?: 'default' | 'success' | 'warning' | 'danger'
}> = ({ label, value, icon, variant = 'default' }) => {
  const bgColors = {
    default: 'bg-slate-800',
    success: 'bg-green-900/20 border-green-800',
    warning: 'bg-amber-900/20 border-amber-800',
    danger: 'bg-red-900/20 border-red-800',
  }

  return (
    <div className={`card ${bgColors[variant]} border-opacity-50`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="metric-label">{label}</p>
          <p className="metric-value mt-2">{value}</p>
        </div>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>
    </div>
  )
}

export const RiskBadge: React.FC<{ level: 'critical' | 'high' | 'medium' | 'low' }> = ({
  level,
}) => {
  const styles = {
    critical: 'bg-red-900 text-red-300 border-red-700',
    high: 'bg-orange-900 text-orange-300 border-orange-700',
    medium: 'bg-yellow-900 text-yellow-300 border-yellow-700',
    low: 'bg-blue-900 text-blue-300 border-blue-700',
  }

  return <div className={`badge ${styles[level]} uppercase text-xs`}>{level}</div>
}

export const DomainBadge: React.FC<{ domain: string }> = ({ domain }) => {
  const domainColors: Record<string, string> = {
    github: 'bg-slate-700 text-slate-200 border-slate-600',
    aws: 'bg-orange-900 text-orange-300 border-orange-700',
    iam: 'bg-purple-900 text-purple-300 border-purple-700',
    network: 'bg-blue-900 text-blue-300 border-blue-700',
    pentest: 'bg-cyan-900 text-cyan-300 border-cyan-700',
  }

  return (
    <div className={`badge ${domainColors[domain] || domainColors.github}`}>
      {domain.toUpperCase()}
    </div>
  )
}

export const StatusIndicator: React.FC<{
  status: 'online' | 'offline' | 'warning'
  label?: string
}> = ({ status, label }) => {
  const colors = {
    online: 'bg-green-500',
    offline: 'bg-red-500',
    warning: 'bg-amber-500',
  }

  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${colors[status]} animate-pulse`} />
      {label && <span className="text-xs text-slate-400">{label}</span>}
    </div>
  )
}

export const TimelineEvent: React.FC<{
  step: number
  title: string
  status: 'completed' | 'current' | 'pending'
  timestamp?: string
}> = ({ step, title, status, timestamp }) => {
  const statusColors = {
    completed: 'bg-green-900 border-green-700 text-green-300',
    current: 'bg-amber-900 border-amber-700 text-amber-300 animate-pulse',
    pending: 'bg-slate-700 border-slate-600 text-slate-400',
  }

  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div
          className={`w-10 h-10 rounded-full border-2 flex items-center justify-center font-semibold ${statusColors[status]}`}
        >
          {step}
        </div>
        {step < 5 && (
          <div
            className={`w-1 h-12 ${
              status === 'completed'
                ? 'bg-green-900'
                : status === 'current'
                  ? 'bg-amber-900'
                  : 'bg-slate-700'
            }`}
          />
        )}
      </div>
      <div className="py-1">
        <h4 className="font-semibold text-slate-100">{title}</h4>
        {timestamp && <p className="text-xs text-slate-500 mt-1">{timestamp}</p>}
      </div>
    </div>
  )
}
