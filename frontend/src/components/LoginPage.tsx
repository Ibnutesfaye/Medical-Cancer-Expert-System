import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface LoginPageProps {
  onLogin: (token: string) => void
  onRegister?: () => void
}

export default function LoginPage({ onLogin, onRegister }: LoginPageProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password.trim()) {
      setError('Please enter username and password.')
      return
    }
    setError('')
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `Login failed (${res.status})`)
      onLogin(data.access_token)
    } catch (err: any) {
      if (err.message === 'Failed to fetch' || err.name === 'TypeError') {
        setError('Backend is not running. Open a terminal and run: venv\\Scripts\\python.exe main.py')
      } else {
        setError(err.message || 'Login failed. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#0f0f0f] flex items-center justify-center p-4 transition-colors duration-300"
      style={{ fontFamily: 'Inter, sans-serif' }}>
      <div className="w-full max-w-sm">

        {/* Logo */}
        <div className="text-center mb-8">
          <img src="/insa-logo.webp" alt="Logo" className="w-14 h-14 mx-auto mb-4 object-contain" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">MedAssist</h1>
          <p className="text-gray-500 dark:text-[#666] text-sm mt-1">Medical Cancer Expert System</p>
        </div>

        {/* Card */}
        <div className="bg-white dark:bg-[#161616] border border-gray-200 dark:border-[#2a2a2a] rounded-2xl p-6 shadow-2xl transition-all duration-300">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-5 transition-colors duration-300">Sign In</h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-[#888] mb-1.5 transition-colors duration-300">Username</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="Enter username"
                autoComplete="username"
                className="w-full px-3.5 py-2.5 bg-gray-50 dark:bg-[#1a1a1a] border border-gray-200 dark:border-[#2a2a2a] rounded-xl text-gray-900 dark:text-white text-sm placeholder-gray-400 dark:placeholder-[#444] focus:outline-none focus:border-blue-500 dark:focus:border-[#4f8ef7] transition-all duration-300"
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-[#888] mb-1.5 transition-colors duration-300">Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Enter password"
                  autoComplete="current-password"
                  className="w-full px-3.5 py-2.5 pr-10 bg-gray-50 dark:bg-[#1a1a1a] border border-gray-200 dark:border-[#2a2a2a] rounded-xl text-gray-900 dark:text-white text-sm placeholder-gray-400 dark:placeholder-[#444] focus:outline-none focus:border-blue-500 dark:focus:border-[#4f8ef7] transition-all duration-300"
                />
                <button type="button" onClick={() => setShowPassword(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 dark:text-[#555] hover:text-gray-600 dark:hover:text-[#aaa] transition-colors">
                  {showPassword ? (
                    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  ) : (
                    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="px-3.5 py-2.5 rounded-xl bg-red-50 dark:bg-[#2a1a1a] border border-red-100 dark:border-[#5a2a2a] text-red-600 dark:text-red-400 text-xs transition-colors duration-300">
                {error}
              </div>
            )}

            {/* Submit */}
            <button type="submit" disabled={loading}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-[#4f8ef7] to-[#7c3aed] text-white text-sm font-semibold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-500/20 flex items-center justify-center gap-2">
              {loading ? (
                <>
                  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Signing in...
                </>
              ) : 'Sign In'}
            </button>
          </form>

          {/* Hint */}
          <div className="mt-4 pt-4 border-t border-gray-100 dark:border-[#2a2a2a] transition-colors duration-300">
            {onRegister && (
              <p className="text-center mt-3 text-xs text-gray-500 dark:text-[#555]">
                No account?{' '}
                <button onClick={onRegister} className="text-blue-600 dark:text-[#4f8ef7] hover:text-blue-700 dark:hover:text-[#7c9ef7] transition-colors font-medium">
                  Create one
                </button>
              </p>
            )}
          </div>
        </div>

        <p className="text-center text-xs text-gray-400 dark:text-[#444] mt-5 transition-colors duration-300">
          Powered by AI · RAG · ResNet18
        </p>
      </div>
    </div>
  )
}
