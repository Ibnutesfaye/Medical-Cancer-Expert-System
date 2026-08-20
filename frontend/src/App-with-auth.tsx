import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider } from './hooks/useAuth'
import ProtectedRoute from './components/ProtectedRoute'
import Login from './components/Login'
import ChatInterface from './components/ChatInterface'
import AdminPage from './components/AdminPage'
import { useState } from 'react'

function MainApp() {
  const [showAdmin, setShowAdmin] = useState(false)
  const token = localStorage.getItem('auth_token') || ''

  if (showAdmin) {
    return <AdminPage token={token} onBack={() => setShowAdmin(false)} />
  }

  return <ChatInterface token={token} onAdminClick={() => setShowAdmin(true)} />
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <MainApp />
              </ProtectedRoute>
            }
          />
          <Route
            path="/settings"
            element={
              <ProtectedRoute>
                <div className="min-h-screen bg-[#0f0f0f] text-white flex items-center justify-center">
                  <div className="text-center">
                    <h1 className="text-2xl font-bold mb-4">Settings</h1>
                    <p className="text-[#666]">Settings page coming soon...</p>
                  </div>
                </div>
              </ProtectedRoute>
            }
          />
          <Route
            path="/help"
            element={
              <ProtectedRoute>
                <div className="min-h-screen bg-[#0f0f0f] text-white flex items-center justify-center">
                  <div className="text-center">
                    <h1 className="text-2xl font-bold mb-4">Help & Support</h1>
                    <p className="text-[#666]">Help documentation coming soon...</p>
                  </div>
                </div>
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App
