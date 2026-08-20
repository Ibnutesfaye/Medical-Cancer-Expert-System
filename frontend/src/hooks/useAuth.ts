const API_URL = import.meta.env.VITE_API_URL || window.location.origin

export function useAuth() {
  const logout = () => {
    localStorage.removeItem('auth_token')
    localStorage.removeItem('conversation_history')
    localStorage.removeItem('chat_sessions')
  }

  const getToken = (): string | null => {
    const token = localStorage.getItem('auth_token')
    if (!token) return null
    try {
      const payload = JSON.parse(atob(token.split('.')[1]))
      if (payload.exp * 1000 < Date.now()) {
        localStorage.removeItem('auth_token')
        return null
      }
      return token
    } catch {
      return null
    }
  }

  const login = async (username: string, password: string): Promise<string> => {
    const res = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `Login failed (${res.status})`)
    }
    const data = await res.json()
    localStorage.setItem('auth_token', data.access_token)
    return data.access_token
  }

  return { login, logout, getToken }
}
