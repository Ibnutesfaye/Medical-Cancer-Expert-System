import { useState, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'
import AdminPage from './components/AdminPage'
import LoginPage from './components/LoginPage'
import RegisterPage from './components/RegisterPage'
import SettingsModal from './components/SettingsModal'
import HelpModal from './components/HelpModal'
import BenchmarkDashboard from './components/BenchmarkDashboard'
import DoctorDashboard from './components/DoctorDashboard'
import ResearchDashboard from './components/ResearchDashboard'

type View = 'login' | 'register' | 'chat' | 'admin' | 'benchmark' | 'doctor' | 'research'

function App() {
  const [token, setToken] = useState<string | null>(null)
  const [view, setView] = useState<View>('login')
  const [showSettings, setShowSettings] = useState(false)
  const [showHelp, setShowHelp] = useState(false)

  const [isAdmin, setIsAdmin] = useState(false)
  const [username, setUsername] = useState('User')

  // On mount: restore valid saved token
  useEffect(() => {
    const saved = localStorage.getItem('auth_token')
    if (saved) {
      try {
        const payload = JSON.parse(atob(saved.split('.')[1]))
        if (payload.exp * 1000 > Date.now()) {
          setToken(saved)
          setIsAdmin(!!payload.is_admin)
          setUsername(payload.sub || 'User')
          setView('chat')
          return
        }
      } catch { }
    }
    setView('login')
  }, [])

  const handleLogin = (newToken: string) => {
    localStorage.setItem('auth_token', newToken)
    setToken(newToken)
    try {
      const payload = JSON.parse(atob(newToken.split('.')[1]))
      setIsAdmin(!!payload.is_admin)
      setUsername(payload.sub || 'User')
    } catch { }
    setView('chat')
  }

  const handleLogout = () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('conversation_history')
    localStorage.removeItem('chat_sessions')
    setToken(null)
    setView('login')
  }

  // Login / Register screens
  if (view === 'register') {
    return (
      <RegisterPage
        onRegistered={() => setView('login')}
        onBack={() => setView('login')}
      />
    )
  }

  if (view === 'login' || !token) {
    return <LoginPage onLogin={handleLogin} onRegister={() => setView('register')} />
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#0f0f0f] transition-colors duration-300">
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
      {showHelp && <HelpModal onClose={() => setShowHelp(false)} isAdmin={isAdmin} />}

      {view === 'admin' && isAdmin ? (
        <AdminPage token={token} onBack={() => setView('chat')} />
      ) : view === 'benchmark' ? (
        <div>
          <div className="flex items-center gap-3 px-6 py-4 bg-white dark:bg-[#161616] border-b border-gray-200 dark:border-[#2a2a2a]">
            <button onClick={() => setView('chat')} className="text-gray-400 hover:text-gray-600 dark:text-[#666] dark:hover:text-[#aaa]">
              <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
            <span className="text-sm font-semibold text-gray-700 dark:text-white">Benchmark Dashboard</span>
          </div>
          <BenchmarkDashboard token={token!} />
        </div>
      ) : view === 'doctor' ? (
        <DoctorDashboard token={token!} onBack={() => setView('chat')} />
      ) : view === 'research' ? (
        <ResearchDashboard token={token!} onBack={() => setView('chat')} />
      ) : (
        <ChatInterface
          token={token}
          username={username}
          onLogout={handleLogout}
          onAdminClick={isAdmin ? () => setView('admin') : undefined}
          onBenchmarkClick={() => setView('benchmark')}
          onDoctorClick={() => setView('doctor')}
          onResearchClick={() => setView('research')}
          onSettings={() => setShowSettings(true)}
          onHelp={() => setShowHelp(true)}
        />
      )}
    </div>
  )
}

export default App
