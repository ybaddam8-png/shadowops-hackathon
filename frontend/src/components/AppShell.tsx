import React from 'react'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

interface AppShellProps {
  currentPage: string
  onPageChange: (page: string) => void
  children: React.ReactNode
}

export const AppShell: React.FC<AppShellProps> = ({
  currentPage,
  onPageChange,
  children,
}) => {
  return (
    <div className="min-h-screen bg-slate-900">
      <Sidebar currentPage={currentPage} onPageChange={onPageChange} />
      <TopBar />
      <main className="ml-56 mt-16 p-8">
        <div className="max-w-7xl">{children}</div>
      </main>
    </div>
  )
}
