import React from 'react'
import {
  LayoutDashboard,
  Zap,
  AlertTriangle,
  BarChart3,
  FileText,
  LogOut,
} from 'lucide-react'

interface SidebarProps {
  currentPage: string
  onPageChange: (page: string) => void
}

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onPageChange }) => {
  const menuItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'mission-control', label: 'Mission Control', icon: Zap },
    { id: 'quarantine', label: 'Quarantine', icon: AlertTriangle },
    { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
    { id: 'incident-report', label: 'Incident Report', icon: FileText },
  ]

  return (
    <div className="fixed left-0 top-0 h-screen w-56 bg-slate-900 border-r border-slate-700 flex flex-col">
      {/* Logo */}
      <div className="px-6 py-8 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-cyan-600 flex items-center justify-center">
            <span className="text-white font-bold text-sm">S</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">ShadowOps</h1>
            <p className="text-xs text-slate-500">AI Supervisor</p>
          </div>
        </div>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        {menuItems.map((item) => {
          const Icon = item.icon
          const isActive = currentPage === item.id
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 text-sm font-medium ${
                isActive
                  ? 'bg-cyan-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-slate-100'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.label}</span>
            </button>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-slate-700">
        <button className="w-full flex items-center gap-3 px-4 py-2 rounded-lg text-slate-400 hover:text-slate-200 text-sm font-medium transition-colors">
          <LogOut className="w-5 h-5" />
          <span>Exit Demo</span>
        </button>
      </div>
    </div>
  )
}
