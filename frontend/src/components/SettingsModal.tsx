import { useTheme } from './ThemeContext';

interface Props { onClose: () => void }

export default function SettingsModal({ onClose }: Props) {
  const { theme, setTheme } = useTheme();

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 transition-all duration-300" onClick={onClose}>
      <div className="bg-white dark:bg-[#161616] border border-gray-200 dark:border-[#2a2a2a] rounded-2xl p-6 max-w-md w-full shadow-2xl transition-all duration-300"
        onClick={e => e.stopPropagation()} style={{ fontFamily: 'Inter, sans-serif' }}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Settings</h2>
          <button onClick={onClose} className="text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#aaa] transition-colors">
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b border-gray-100 dark:border-[#2a2a2a]">
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">Theme</p>
              <p className="text-xs text-gray-500 dark:text-[#666] mt-0.5">Choose your appearance</p>
            </div>
            <div className="flex bg-gray-100 dark:bg-[#242424] p-1 rounded-xl">
              <button 
                onClick={() => setTheme('light')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${theme === 'light' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 dark:text-[#aaa] hover:text-gray-700 dark:hover:text-[#ccc]'}`}
              >
                Light
              </button>
              <button 
                onClick={() => setTheme('dark')}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${theme === 'dark' ? 'bg-gray-700 dark:bg-[#333] text-white shadow-sm' : 'text-gray-500 dark:text-[#aaa] hover:text-gray-700 dark:hover:text-[#ccc]'}`}
              >
                Dark
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between py-3 border-b border-gray-100 dark:border-[#2a2a2a]">
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">Language</p>
              <p className="text-xs text-gray-500 dark:text-[#666] mt-0.5">Interface language</p>
            </div>
            <span className="px-3 py-1 rounded-full bg-gray-100 dark:bg-[#242424] text-gray-500 dark:text-[#aaa] text-xs">English</span>
          </div>

          <div className="flex items-center justify-between py-3 border-b border-gray-100 dark:border-[#2a2a2a]">
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">Backend</p>
              <p className="text-xs text-gray-500 dark:text-[#666] mt-0.5 truncate max-w-[180px]">{import.meta.env.VITE_API_URL || window.location.origin}</p>
            </div>
            <span className="px-3 py-1 rounded-full bg-green-50 dark:bg-[#1a3a2a] text-green-600 dark:text-[#4ade80] text-xs font-medium">Connected</span>
          </div>

          <div className="flex items-center justify-between py-3">
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">Version</p>
              <p className="text-xs text-gray-500 dark:text-[#666] mt-0.5">Medical Cancer Expert System</p>
            </div>
            <span className="px-3 py-1 rounded-full bg-gray-100 dark:bg-[#242424] text-gray-500 dark:text-[#aaa] text-xs">v2.0.0</span>
          </div>
        </div>

        <button onClick={onClose}
          className="w-full mt-6 py-2.5 rounded-xl bg-gray-100 dark:bg-[#242424] text-gray-600 dark:text-[#ccc] hover:bg-gray-200 dark:hover:bg-[#2e2e2e] transition-colors text-sm font-medium">
          Close
        </button>
      </div>
    </div>
  )
}
