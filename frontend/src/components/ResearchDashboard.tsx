import { useState, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface ResearchStats {
  total_analyses:  number
  cancer_detected: number
  healthy:         number
  total_patients:  number
  detection_rate:  number
}

interface BenchmarkResult {
  id: number
  inference_mode:      string
  inference_time_ms:   number | null
  encryption_time_ms:  number | null
  memory_mb:           number | null
  security_level_bits: number
  privacy_level:       string | null
  cancer_detected:     boolean | null
  confidence:          number | null
  created_at:          string
}

interface ResearchDashboardProps {
  token: string
  onBack: () => void
}

const MODE_COLORS: Record<string, string> = {
  standard:    'text-blue-600 dark:text-blue-400',
  concrete_ml: 'text-purple-600 dark:text-purple-400',
  openfhe:     'text-green-600 dark:text-green-400',
}

const MODE_ICONS: Record<string, string> = {
  standard:    '⚡',
  concrete_ml: '🔐',
  openfhe:     '🛡️',
}

export default function ResearchDashboard({ token, onBack }: ResearchDashboardProps) {
  const [stats, setStats]         = useState<ResearchStats | null>(null)
  const [benchmarks, setBenchmarks] = useState<BenchmarkResult[]>([])
  const [modeStats, setModeStats] = useState<Record<string, any>>({})
  const [loading, setLoading]     = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'benchmarks' | 'privacy'>('overview')

  const h = { Authorization: `Bearer ${token}` }

  useEffect(() => { loadAll() }, [])

  const loadAll = async () => {
    setLoading(true)
    try {
      const [statsRes, benchRes, modeRes] = await Promise.all([
        fetch(`${API_URL}/doctor/dashboard/stats`, { headers: h }),
        fetch(`${API_URL}/benchmark/results?limit=50`, { headers: h }),
        fetch(`${API_URL}/benchmark/stats`, { headers: h }),
      ])
      if (statsRes.ok)  setStats(await statsRes.json())
      if (benchRes.ok)  setBenchmarks(await benchRes.json())
      if (modeRes.ok)   setModeStats(await modeRes.json())
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  const formatDate = (d: string) => new Date(d).toLocaleDateString()

  const STATIC_METRICS = [
    { label: 'Brain Cancer Accuracy',    value: '88.75%', mode: 'standard',    icon: '🧠' },
    { label: 'Lung Cancer Accuracy',     value: '86.36%', mode: 'standard',    icon: '🫁' },
    { label: 'Skin Cancer Accuracy',     value: '89.52%', mode: 'standard',    icon: '🩺' },
    { label: 'FHE Accuracy (CML)',       value: '~84%',   mode: 'concrete_ml', icon: '🔐' },
    { label: 'Privacy Level (CML)',      value: '128-bit', mode: 'concrete_ml', icon: '🔒' },
    { label: 'Privacy Level (OpenFHE)',  value: '128-bit', mode: 'openfhe',     icon: '🛡️' },
  ]

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-gray-50 dark:bg-[#0f0f0f] text-gray-900 dark:text-white">

      {/* Header */}
      <header className="bg-white dark:bg-[#161616] border-b border-gray-200 dark:border-[#2a2a2a] px-6 py-4 flex items-center gap-4 flex-shrink-0">
        <button onClick={onBack} className="text-gray-400 hover:text-gray-600 dark:text-[#666] dark:hover:text-[#aaa]">
          <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <h1 className="text-lg font-bold">🔬 Research Dashboard</h1>
        <button onClick={loadAll}
          className="ml-auto text-xs px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-[#242424] text-gray-600 dark:text-[#aaa] hover:bg-gray-200 dark:hover:bg-[#2e2e2e] transition-colors">
          ↻ Refresh
        </button>
      </header>

      {/* Tabs */}
      <div className="bg-white dark:bg-[#161616] border-b border-gray-100 dark:border-[#2a2a2a] px-6">
        <div className="flex gap-6">
          {([['overview','Overview'],['benchmarks','Benchmarks'],['privacy','Privacy']] as const).map(([tab, label]) => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? 'border-purple-500 text-gray-900 dark:text-white'
                  : 'border-transparent text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#aaa]'
              }`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">

          {loading && (
            <div className="text-center py-12 text-gray-400 dark:text-[#666]">Loading research data...</div>
          )}

          {/* Overview tab */}
          {!loading && activeTab === 'overview' && (
            <>
              {/* System stats */}
              {stats && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {[
                    { label: 'Total Analyses',  val: stats.total_analyses,  color: 'text-blue-600' },
                    { label: 'Cancer Detected', val: stats.cancer_detected,  color: 'text-red-600' },
                    { label: 'Healthy',          val: stats.healthy,          color: 'text-green-600' },
                    { label: 'Total Patients',   val: stats.total_patients,   color: 'text-purple-600' },
                    { label: 'Detection Rate',   val: `${stats.detection_rate}%`, color: 'text-orange-600' },
                    { label: 'Benchmark Runs',   val: benchmarks.length,      color: 'text-teal-600' },
                  ].map(({ label, val, color }) => (
                    <div key={label} className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-4 shadow-sm">
                      <p className="text-xs text-gray-400 dark:text-[#666] mb-1">{label}</p>
                      <p className={`text-2xl font-bold ${color}`}>{val}</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Static accuracy metrics */}
              <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5">
                <h2 className="text-sm font-semibold mb-4">Model Performance Metrics</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {STATIC_METRICS.map(m => (
                    <div key={m.label} className="p-3 rounded-lg bg-gray-50 dark:bg-[#1a1a1a] border border-gray-100 dark:border-[#2a2a2a]">
                      <div className="flex items-center gap-2 mb-1">
                        <span>{m.icon}</span>
                        <span className={`text-xs font-semibold uppercase ${MODE_COLORS[m.mode] || 'text-gray-500'}`}>
                          {m.mode.replace('_', ' ')}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 dark:text-[#888]">{m.label}</p>
                      <p className="text-base font-bold text-gray-800 dark:text-white mt-0.5">{m.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Per-mode aggregate stats from DB */}
              {Object.keys(modeStats).length > 0 && (
                <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5">
                  <h2 className="text-sm font-semibold mb-4">Benchmark Aggregate (from DB)</h2>
                  <div className="grid grid-cols-3 gap-4">
                    {(['standard','concrete_ml','openfhe'] as const).map(mode => {
                      const s = modeStats[mode]
                      if (!s) return null
                      return (
                        <div key={mode} className="text-center p-3 rounded-lg bg-gray-50 dark:bg-[#1a1a1a]">
                          <p className="text-lg mb-1">{MODE_ICONS[mode]}</p>
                          <p className={`text-xs font-semibold capitalize mb-2 ${MODE_COLORS[mode]}`}>
                            {mode.replace('_', ' ')}
                          </p>
                          <p className="text-xs text-gray-400 dark:text-[#666]">Avg inference</p>
                          <p className="text-sm font-bold">{s.avg_inference_ms > 0 ? `${s.avg_inference_ms}ms` : '—'}</p>
                          <p className="text-xs text-gray-400 dark:text-[#666] mt-1">Runs: {s.total_runs}</p>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </>
          )}

          {/* Benchmarks tab */}
          {!loading && activeTab === 'benchmarks' && (
            <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-100 dark:border-[#2a2a2a]">
                <h2 className="text-sm font-semibold">All Benchmark Results ({benchmarks.length})</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 dark:bg-[#1a1a1a]">
                    <tr>
                      {['Mode','Inference (ms)','Encrypt (ms)','Memory (MB)','Security','Privacy','Cancer','Date'].map(h => (
                        <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 dark:text-[#888] uppercase">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarks.length === 0
                      ? <tr><td colSpan={8} className="px-4 py-8 text-center text-gray-400 dark:text-[#666]">No benchmarks yet. Upload an image in Encrypted AI to run one.</td></tr>
                      : benchmarks.map(r => (
                        <tr key={r.id} className="border-t border-gray-50 dark:border-[#2a2a2a] hover:bg-gray-50 dark:hover:bg-[#1a1a1a]">
                          <td className="px-4 py-2">
                            <span className={`text-xs font-semibold ${MODE_COLORS[r.inference_mode] || ''}`}>
                              {MODE_ICONS[r.inference_mode] || ''} {r.inference_mode?.replace('_',' ')}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.inference_time_ms ?? '—'}</td>
                          <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.encryption_time_ms ?? '—'}</td>
                          <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.memory_mb?.toFixed(1) ?? '—'}</td>
                          <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.security_level_bits ? `${r.security_level_bits}-bit` : 'None'}</td>
                          <td className="px-4 py-2 text-gray-500 dark:text-[#888] text-xs">{r.privacy_level || '—'}</td>
                          <td className="px-4 py-2">
                            {r.cancer_detected === null ? <span className="text-gray-400 text-xs">—</span>
                              : r.cancer_detected
                              ? <span className="px-2 py-0.5 rounded-full text-xs bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400">Yes</span>
                              : <span className="px-2 py-0.5 rounded-full text-xs bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400">No</span>
                            }
                          </td>
                          <td className="px-4 py-2 text-gray-500 dark:text-[#888] text-xs">{formatDate(r.created_at)}</td>
                        </tr>
                      ))
                    }
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Privacy tab */}
          {!loading && activeTab === 'privacy' && (
            <div className="space-y-4">
              <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5">
                <h2 className="text-sm font-semibold mb-4">Privacy Technology Comparison</h2>
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 dark:bg-[#1a1a1a]">
                    <tr>
                      {['Feature','Standard AI','Concrete ML','OpenFHE'].map(h => (
                        <th key={h} className="px-4 py-2 text-left text-xs font-semibold text-gray-500 dark:text-[#888]">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50 dark:divide-[#2a2a2a]">
                    {[
                      ['Patient image privacy', '❌ None',      '✅ Feature-level', '✅ Pixel-level'],
                      ['FHE encryption',        '❌ No',         '✅ 128-bit',        '✅ 128-bit (CKKS)'],
                      ['HIPAA compatible',       '⚠️  Needs controls', '✅ Yes',       '✅ Yes'],
                      ['GDPR compliant',         '⚠️  Needs controls', '✅ Yes',       '✅ Yes'],
                      ['Server sees raw data',   '✅ Yes (risk)', '❌ No',             '❌ No'],
                      ['Inference speed',        '~50ms',        '~2–10s',           '~2–5 min'],
                      ['Accuracy vs baseline',   '100%',         '~95%',             '~93%'],
                      ['GPU support',            '✅ Yes',        '❌ No',             '❌ No'],
                      ['Research ready',         '✅ Yes',        '✅ Yes',            '✅ Yes'],
                    ].map(([feature, std, cml, fhe]) => (
                      <tr key={feature} className="hover:bg-gray-50 dark:hover:bg-[#1a1a1a]">
                        <td className="px-4 py-2 font-medium text-gray-700 dark:text-[#ccc]">{feature}</td>
                        <td className="px-4 py-2 text-gray-600 dark:text-[#aaa]">{std}</td>
                        <td className="px-4 py-2 text-gray-600 dark:text-[#aaa]">{cml}</td>
                        <td className="px-4 py-2 text-gray-600 dark:text-[#aaa]">{fhe}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5">
                <h2 className="text-sm font-semibold mb-3">Differential Privacy Status</h2>
                <p className="text-xs text-gray-500 dark:text-[#888] mb-3">
                  Run <code className="bg-gray-100 dark:bg-[#242424] px-1.5 py-0.5 rounded text-xs">python -m ai.dp_training</code> to train with DP-SGD.
                </p>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { label: 'Target ε', val: '8.0', desc: 'Privacy budget' },
                    { label: 'Target δ', val: '1e-5', desc: 'Failure probability' },
                    { label: 'Grad clip', val: '1.0', desc: 'Max gradient norm' },
                  ].map(({ label, val, desc }) => (
                    <div key={label} className="p-3 rounded-lg bg-gray-50 dark:bg-[#1a1a1a] border border-gray-100 dark:border-[#2a2a2a] text-center">
                      <p className="text-xs text-gray-400 dark:text-[#666]">{desc}</p>
                      <p className="text-sm font-bold text-purple-600 dark:text-purple-400 mt-0.5">{label} = {val}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
