import { useState } from 'react'
import { AppShell } from './components/AppShell'
import { Overview } from './pages/Overview'
import { MissionControl } from './pages/MissionControl'
import { Quarantine } from './pages/Quarantine'
import { Benchmarks } from './pages/Benchmarks'
import { IncidentReport } from './pages/IncidentReport'

function App() {
  const [currentPage, setCurrentPage] = useState('overview')

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <Overview />
      case 'mission-control':
        return <MissionControl onNavigate={setCurrentPage} />
      case 'quarantine':
        return <Quarantine />
      case 'benchmarks':
        return <Benchmarks />
      case 'incident-report':
        return <IncidentReport onNavigate={setCurrentPage} />
      default:
        return <Overview />
    }
  }

  return (
    <AppShell currentPage={currentPage} onPageChange={setCurrentPage}>
      {renderPage()}
    </AppShell>
  )
}

export default App
