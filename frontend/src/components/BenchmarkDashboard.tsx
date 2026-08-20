import { useState, useEffect, useRef } from 'react'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface ModeResult {
  inference_time_ms: number | null
  encryption_time_ms: number | null
  memory_mb: number | null
  security_level_bits: number
  privacy_level: string | null
  confidence: number | null
  cancer_detected: boolean | null
  cancer_type: string | null
  error: string | null
}

interface BenchmarkResult {
  timestamp: string
  results: {
    standard:    ModeResult | null
    concrete_ml: ModeResult | null
    openfhe:     ModeResult | null
  }
  summary: {
    fastest_mode:    string
    most_private:    string
    best_balance:    string
    accuracy_winner: string
  }
}

interface BenchmarkDashboardProps {
  token: string
}

const MODES = ['standard', 'concrete_ml', 'openfhe'] as const
const MODE_LABELS: Record<string, string> = {
  standard:    '⚡ Standard AI',
  concrete_ml: '🔐 Concrete ML',
  openfhe:     '🛡️ OpenFHE',
}
const MODE_COLORS: Record<string, string> = {
  standard:    'bg-blue-500',
  concrete_ml: 'bg-purple-500',
  openfhe:     'bg-green-500',
}

function Bar({ value, max, color }: { value: number | null, max: number, color: string }) {
  if (value == null || value < 0) return <div className="h-3 bg-gray-100 dark:bg-[#333] rounded-full w-full" />
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="h-3 bg-gray-100 dark:bg-[#333] rounded-full w-full">
      <div className={`h-3 rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export default function BenchmarkDashboard({ token }: BenchmarkDashboardProps) {
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [history, setHistory] = useState<any[]>([])
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<'live' | 'history'>('live')
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { fetchHistory() }, [])

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/benchmark/results?limit=20`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (res.ok) setHistory(await res.json())
    } catch { /* ignore */ }
  }

  const handleBenchmark = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setRunning(true)
    setError('')
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_URL}/inference/benchmark/quick`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      })
      if (!res.ok) throw new Error('Benchmark failed')
      const data: BenchmarkResult = await res.json()
      setResult(data)
      // Save each mode result to DB
      for (const mode of MODES) {
        const r = data.results[mode]
        if (r) {
          await fetch(`${API_URL}/benchmark/save`, {
            method: 'POST',
            headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({ inference_mode: mode, ...r }),
          })
        }
      }
      fetchHistory()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRunning(false)
      e.target.value = ''
    }
  }

  const maxTime = result
    ? Math.max(...MODES.map(m => result.results[m]?.inference_time_ms ?? 0))
    : 1

  const staticComparison = [
    { metric: 'Accuracy',       standard: '88.75%', concrete_ml: '~84–86%',  openfhe: '~82–85%' },
    { metric: 'Precision',      standard: '96.70%', concrete_ml: '~91%',     openfhe: '~89%' },
    { metric: 'Recall',         standard: '88.00%', concrete_ml: '~84%',     openfhe: '~82%' },
    { metric: 'F1 Score',       standard: '92.15%', concrete_ml: '~87%',     openfhe: '~85%' },
    { metric: 'Latency',        standard: '~50ms',  concrete_ml: '~2–10s',   openfhe: '~2–5 min' },
    { metric: 'Privacy',        standard: 'None',   concrete_ml: 'Feature FHE', openfhe: 'Pixel FHE' },
    { metric: 'Security',       standard: '0-bit',  concrete_ml: '128-bit',  openfhe: '128-bit' },
    { metric: 'GPU support',    standard: '✅',     concrete_ml: '❌',        openfhe: '❌' },
    { metric: 'HIPAA Ready',    standard: '⚠️',     concrete_ml: '✅',        openfhe: '✅' },
  ]

  return (
    <div className="h-screen w-screen overflow-y-auto bg-gray-50 dark:bg-[#0f0f0f] text-gray-900 dark:text-white">
      <div className="max-w-5xl mx-auto px-6 py-8">

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold">📊 Inference Mode Benchmark</h1>
            <p className="text-sm text-gray-500 dark:text-[#666] mt-1">
              Compare Standard AI vs Concrete ML (FHE) vs OpenFHE side-by-side
            </p>
          </div>
          <label className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium cursor-pointer transition-all">
            {running ? '⏳ Running...' : '▶ Run Benchmark'}
            <input ref={fileRef} type="file" accept="image/*" className="hidden"
              onChange={handleBenchmark} disabled={running} />
          </label>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-6 border-b border-gray-200 dark:border-[#2a2a2a]">
          {(['live', 'history'] as const).map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`pb-2 text-sm font-medium border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-purple-500 text-gray-900 dark:text-white'
                  : 'border-transparent text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#aaa]'
              }`}>
              {tab === 'live' ? '🔴 Live Results' : '📋 History'}
            </button>
          ))}
        </div>

        {error && (
          <div className="mb-4 px-4 py-3 rounded-xl bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-900/20 text-red-600 dark:text-red-400 text-sm">
            {error}
          </div>
        )}

        {activeTab === 'live' && (
          <div className="space-y-6">

            {/* Live result bars */}
            {result && (
              <div className="bg-white dark:bg-[#161616] rounded-2xl border border-gray-100 dark:border-[#2a2a2a] p-6 space-y-5">
                <h2 className="text-sm font-semibold text-gray-700 dark:text-[#ccc]">Inference Time Comparison</h2>
                {MODES.map(mode => {
                  const r = result.results[mode]
                  const ms = r?.inference_time_ms
                  return (
                    <div key={mode}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-medium">{MODE_LABELS[mode]}</span>
                        <span className="text-gray-500 dark:text-[#888]">
                          {ms != null && ms >= 0 ? `${ms.toLocaleString()}ms` : r?.error ? '❌ N/A' : '—'}
                        </span>
                      </div>
                      <Bar value={ms ?? null} max={maxTime || 1} color={MODE_COLORS[mode]} />
                      {r?.error && (
                        <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-1">{r.error}</p>
                      )}
                    </div>
                  )
                })}

                {/* Summary badges */}
                <div className="flex flex-wrap gap-2 pt-2">
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300">
                    ⚡ Fastest: {result.summary.fastest_mode}
                  </span>
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300">
                    🔐 Most private: {result.summary.most_private}
                  </span>
                  <span className="px-3 py-1 rounded-full text-xs font-medium bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300">
                    ⚖️ Best balance: {result.summary.best_balance}
                  </span>
                </div>
              </div>
            )}

            {/* Static comparison table */}
            <div className="bg-white dark:bg-[#161616] rounded-2xl border border-gray-100 dark:border-[#2a2a2a] overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-100 dark:border-[#2a2a2a]">
                <h2 className="text-sm font-semibold text-gray-700 dark:text-[#ccc]">Full Comparison Table</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 dark:bg-[#1a1a1a]">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 dark:text-[#888] uppercase">Metric</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-blue-600 uppercase">⚡ Standard</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-purple-600 uppercase">🔐 Concrete ML</th>
                      <th className="px-4 py-3 text-center text-xs font-semibold text-green-600 uppercase">🛡️ OpenFHE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {staticComparison.map((row, i) => (
                      <tr key={row.metric} className={i % 2 === 0 ? '' : 'bg-gray-50 dark:bg-[#1a1a1a]'}>
                        <td className="px-4 py-2 font-medium text-gray-700 dark:text-[#ccc]">{row.metric}</td>
                        <td className="px-4 py-2 text-center text-gray-600 dark:text-[#aaa]">{row.standard}</td>
                        <td className="px-4 py-2 text-center text-gray-600 dark:text-[#aaa]">{row.concrete_ml}</td>
                        <td className="px-4 py-2 text-center text-gray-600 dark:text-[#aaa]">{row.openfhe}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="bg-white dark:bg-[#161616] rounded-2xl border border-gray-100 dark:border-[#2a2a2a] overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-[#1a1a1a]">
                <tr>
                  {['Mode', 'Inference (ms)', 'Encrypt (ms)', 'Memory (MB)', 'Security', 'Privacy', 'Date'].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 dark:text-[#888] uppercase">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.length === 0
                  ? <tr><td colSpan={7} className="px-4 py-8 text-center text-gray-400 dark:text-[#666]">No benchmark history yet</td></tr>
                  : history.map(r => (
                    <tr key={r.id} className="border-t border-gray-50 dark:border-[#2a2a2a] hover:bg-gray-50 dark:hover:bg-[#1a1a1a]">
                      <td className="px-4 py-2">{MODE_LABELS[r.inference_mode] || r.inference_mode}</td>
                      <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.inference_time_ms ?? '—'}</td>
                      <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.encryption_time_ms ?? '—'}</td>
                      <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.memory_mb ?? '—'}</td>
                      <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.security_level_bits ? `${r.security_level_bits}-bit` : 'None'}</td>
                      <td className="px-4 py-2 text-gray-500 dark:text-[#888]">{r.privacy_level || '—'}</td>
                      <td className="px-4 py-2 text-gray-500 dark:text-[#888] text-xs">
                        {r.created_at ? new Date(r.created_at).toLocaleDateString() : '—'}
                      </td>
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
