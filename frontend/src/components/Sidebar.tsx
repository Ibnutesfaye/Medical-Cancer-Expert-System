import { useState } from 'react'
import { useAuth } from '../hooks/useAuth'

interface ChatHistoryItem {
  id: string
  title: string
  date: string
}

interface SidebarProps {
  open: boolean
  onClose: () => void
  onNewChat: () => void
  chatHistory: ChatHistoryItem[]
  activeId: string | null
  onSelectChat: (id: string) => void
  onDeleteChat: (id: string) => void
  username?: string
  onAdminClick?: () => void
  onBenchmarkClick?: () => void
  onDoctorClick?: () => void
  onResearchClick?: () => void
  onLogout?: () => void
  onSettings?: () => void
  onHelp?: () => void
}

function groupByDate(items: ChatHistoryItem[]) {
  const groups: Record<string, ChatHistoryItem[]> = {}
  items.forEach(item => {
    if (!groups[item.date]) groups[item.date] = []
    groups[item.date].push(item)
  })
  return groups
}

export default function Sidebar({
  open, onClose, onNewChat, chatHistory, activeId,
  onSelectChat, onDeleteChat, username = 'Admin',
  onAdminClick, onBenchmarkClick, onDoctorClick, onResearchClick,
  onLogout, onSettings, onHelp,
}: SidebarProps) {
  const [showMenu, setShowMenu] = useState(false)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false)
  const groups = groupByDate(chatHistory)
  const { logout } = useAuth()

  const handleSelect = (id: string) => { onSelectChat(id); onClose() }
  const handleNewChat = () => { onNewChat(); onClose() }

  const handleLogout = () => {
    setShowLogoutConfirm(false)
    setShowMenu(false)
    logout()
    onLogout?.()
  }

  const handleMenuAction = (action: string) => {
    setShowMenu(false)
    if (action === 'Settings') onSettings?.()
    else if (action === 'Help') onHelp?.()
    else if (action === 'Sign out') setShowLogoutConfirm(true)
  }

  return (
    <>
      {open && <div className="fixed inset-0 bg-black/60 z-30 lg:hidden" onClick={onClose} />}

      {/* Logout Confirm Modal */}
      {showLogoutConfirm && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
          <div className="bg-[#161616] border border-[#2a2a2a] rounded-2xl p-6 max-w-sm w-full shadow-2xl">
            <h3 className="text-lg font-semibold text-white mb-2">Sign Out</h3>
            <p className="text-[#aaa] text-sm mb-6">Are you sure you want to sign out?</p>
            <div className="flex gap-3">
              <button onClick={() => setShowLogoutConfirm(false)}
                className="flex-1 px-4 py-2 rounded-xl bg-[#242424] text-[#ccc] hover:bg-[#2e2e2e] transition-colors text-sm">
                Cancel
              </button>
              <button onClick={handleLogout}
                className="flex-1 px-4 py-2 rounded-xl bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors text-sm">
                Sign Out
              </button>
            </div>
          </div>
        </div>
      )}

      <aside className={[
        'fixed top-0 left-0 h-full z-40 flex flex-col',
        'w-[260px] bg-white dark:bg-[#161616] border-r border-gray-200 dark:border-[#2a2a2a]',
        'transition-all duration-300 ease-in-out shadow-xl lg:shadow-none',
        open ? 'translate-x-0 lg:ml-0' : '-translate-x-full lg:-ml-[260px]',
        'lg:translate-x-0 lg:static lg:z-auto lg:flex-shrink-0',
      ].join(' ')} style={{ fontFamily: 'Inter, sans-serif' }}>

        {/* Logo */}
        <div className="flex items-center gap-3 px-4 pt-5 pb-3">
          <img src="/insa-logo.webp" alt="Logo" className="w-8 h-8 object-contain flex-shrink-0" />
          <span className="text-gray-900 dark:text-white font-bold text-[17px] tracking-tight">MedAssist</span>
          <button onClick={onClose} className="ml-auto text-gray-400 dark:text-[#555] hover:text-gray-600 dark:hover:text-[#aaa] transition-colors lg:hidden">
            <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* New Chat */}
        <div className="px-3 pb-4">
          <button onClick={handleNewChat}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl bg-gray-50 dark:bg-[#242424] border border-gray-200 dark:border-[#333] text-gray-700 dark:text-[#e0e0e0] text-sm font-medium hover:bg-gray-100 dark:hover:bg-[#2e2e2e] transition-colors shadow-sm">
            <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </button>
        </div>

        {/* History */}
        <div className="flex-1 overflow-y-auto px-2">
          {Object.keys(groups).length === 0 && (
            <p className="text-xs text-gray-400 dark:text-[#555] text-center mt-6">No chat history yet</p>
          )}
          {Object.entries(groups).map(([date, items]) => (
            <div key={date} className="mb-4">
              <p className="text-[11px] font-semibold text-gray-400 dark:text-[#555] uppercase tracking-wider px-2 mb-1">{date}</p>
              {items.map(item => (
                <div key={item.id} className="relative group"
                  onMouseEnter={() => setHoveredId(item.id)}
                  onMouseLeave={() => setHoveredId(null)}>
                  <button onClick={() => handleSelect(item.id)}
                    className={['w-full text-left px-2.5 py-2 pr-8 rounded-lg text-sm truncate transition-all block',
                      activeId === item.id 
                        ? 'bg-gray-100 dark:bg-[#242424] text-gray-900 dark:text-white font-medium' 
                        : 'text-gray-600 dark:text-[#aaa] hover:bg-gray-50 dark:hover:bg-[#1e1e1e] hover:text-gray-900 dark:hover:text-[#ccc]'].join(' ')}>
                    {item.title}
                  </button>
                  {(hoveredId === item.id || activeId === item.id) && (
                    <button onClick={e => { e.stopPropagation(); onDeleteChat(item.id) }}
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded-md text-gray-400 dark:text-[#555] hover:text-red-500 hover:bg-red-50 dark:hover:bg-[#2a1a1a] transition-all">
                      <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>

        {/* Bottom */}
        <div className="relative px-3 py-3 border-t border-gray-100 dark:border-[#2a2a2a]">
          {onAdminClick && (
            <button onClick={onAdminClick}
              className="w-full flex items-center gap-2 px-3 py-2 mb-1 rounded-xl bg-blue-50 dark:bg-[#1a3a5f] border border-blue-100 dark:border-[#2a5a8f] text-blue-600 dark:text-[#60a5fa] text-sm font-medium hover:bg-blue-100 dark:hover:bg-[#1e4a7f] transition-colors">
              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Admin Dashboard
            </button>
          )}
          {onDoctorClick && (
            <button onClick={onDoctorClick}
              className="w-full flex items-center gap-2 px-3 py-2 mb-1 rounded-xl bg-teal-50 dark:bg-[#1a3a3a] border border-teal-100 dark:border-[#2a5a5a] text-teal-600 dark:text-[#5eead4] text-sm font-medium hover:bg-teal-100 dark:hover:bg-[#1e4a4a] transition-colors">
              <span className="text-base">👨‍⚕️</span>
              Doctor Dashboard
            </button>
          )}
          {onBenchmarkClick && (
            <button onClick={onBenchmarkClick}
              className="w-full flex items-center gap-2 px-3 py-2 mb-1 rounded-xl bg-purple-50 dark:bg-[#2a1a4a] border border-purple-100 dark:border-[#4a2a7a] text-purple-600 dark:text-[#c084fc] text-sm font-medium hover:bg-purple-100 dark:hover:bg-[#3a2a5a] transition-colors">
              <span className="text-base">📊</span>
              Benchmark
            </button>
          )}
          {onResearchClick && (
            <button onClick={onResearchClick}
              className="w-full flex items-center gap-2 px-3 py-2 mb-2 rounded-xl bg-orange-50 dark:bg-[#2a1a1a] border border-orange-100 dark:border-[#4a2a2a] text-orange-600 dark:text-[#fb923c] text-sm font-medium hover:bg-orange-100 dark:hover:bg-[#3a2a2a] transition-colors">
              <span className="text-base">🔬</span>
              Research
            </button>
          )}

          <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-xl hover:bg-gray-50 dark:hover:bg-[#1e1e1e] transition-colors cursor-pointer"
            onClick={() => setShowMenu(v => !v)}>
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold flex-shrink-0 shadow-sm">
              {username[0].toUpperCase()}
            </div>
            <span className="flex-1 text-sm font-medium text-gray-700 dark:text-[#ccc] truncate">{username}</span>
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20" className="text-gray-400 dark:text-[#666]">
              <path d="M6 10a2 2 0 11-4 0 2 2 0 014 0zm6 0a2 2 0 11-4 0 2 2 0 014 0zm6 0a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>

          {showMenu && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setShowMenu(false)} />
              <div className="absolute bottom-[72px] left-3 right-3 bg-white dark:bg-[#1e1e1e] border border-gray-200 dark:border-[#333] rounded-xl p-1.5 shadow-2xl z-50">
                {[
                  { label: 'Settings', icon: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z' },
                  { label: 'Help', icon: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
                  { label: 'Sign out', icon: 'M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1' },
                ].map(({ label, icon }) => (
                  <button key={label} onClick={() => handleMenuAction(label)}
                    className={['w-full text-left px-3 py-2 rounded-lg text-sm hover:bg-gray-50 dark:hover:bg-[#2a2a2a] transition-colors flex items-center gap-2',
                      label === 'Sign out' ? 'text-red-500' : 'text-gray-700 dark:text-[#ccc]'].join(' ')}>
                    <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d={icon} />
                    </svg>
                    {label}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </aside>
    </>
  )
}
