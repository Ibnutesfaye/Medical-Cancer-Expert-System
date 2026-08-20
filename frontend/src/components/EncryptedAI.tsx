import { useState, useRef } from 'react'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface InferenceMode {
  id: string
  name: string
  description: string
  available: boolean
  status_msg?: string
  privacy: string
  latency_est: string
  security: string
}

interface InferenceResult {
  inference_mode: string
  cancer_detected: boolean | null
  cancer_type: string | null
  confidence: number | null
  inference_time_ms: number
  encryption_time_ms: number
  decryption_time_ms: number
  security_level_bits: number
  privacy_level?: string
  message?: string
  error?: string
  status?: string
}

interface EncryptedAIProps {
  token: string
  onClose: () => void
}

const MODE_ICONS: Record<string, string> = {
  standard:    '⚡',
  concrete_ml: '🔐',
  openfhe:     '🛡️',
}

const MODE_COLORS: Record<string, string> = {
  standard:    'border-blue-200 dark:border-blue-900/40 bg-blue-50 dark:bg-blue-900/10',
  concrete_ml: 'border-purple-200 dark:border-purple-900/40 bg-purple-50 dark:bg-purple-900/10',
  openfhe:     'border-green-200 dark:border-green-900/40 bg-green-50 dark:bg-green-900/10',
}

export default function EncryptedAI({ token, onClose }: EncryptedAIProps) {
  const [modes, setModes] = useState<InferenceMode[]>([])
  const [selectedMode, setSelectedMode] = useState('standard')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [error, setError] = useState('')
  const [modesLoaded, setModesLoaded] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const loadModes = async () => {
    if (modesLoaded) return
    try {
      const res = await fetch(`${API_URL}/inference/modes`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (res.ok) {
        const data = await res.json()
        setModes(data.modes || [])
        setModesLoaded(true)
      }
    } catch { /* silently ignore */ }
  }

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError('')
    setPreview(URL.createObjectURL(f))
    loadModes()
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(
        `${API_URL}/inference/analyze?mode=${selectedMode}`,
        { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: form }
      )

      // Always read raw text first — never call .json() directly
      // This prevents "Unexpected end of JSON input" on empty/error responses
      const text = await res.text()

      if (!text || text.trim() === '') {
        throw new Error('Server returned an empty response. Check backend logs.')
      }

      let data: any
      try {
        data = JSON.parse(text)
      } catch {
        throw new Error(`Server returned invalid JSON: ${text.slice(0, 100)}`)
      }

      if (!res.ok) {
        throw new Error(data?.detail || `Server error ${res.status}`)
      }

      setResult(data as InferenceResult)
    } catch (e: any) {
      setError(e.message || 'Failed to analyze')
    } finally {
      setLoading(false)
    }
  }

  const privacySteps: Record<string, string[]> = {
    standard: [
      '❌ Image uploaded to server in plaintext',
      '⚡ Server runs ResNet18 inference directly',
      '📤 Result returned (no encryption)',
    ],
    concrete_ml: [
      '✅ ResNet18 feature extraction runs locally',
      '🔐 Features encrypted with 128-bit FHE key',
      '📤 Only encrypted vector sent to server',
      '🖥️ Server infers on encrypted data (cannot decrypt)',
      '🔓 Result decrypted on your device',
    ],
    openfhe: [
      '✅ Image pixels encrypted with CKKS scheme (128-bit)',
      '🔐 Entire computation runs on ciphertexts',
      '🖥️ Server never sees pixels OR features',
      '🔓 Encrypted prediction returned and decrypted locally',
    ],
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg bg-white dark:bg-[#161616] border border-gray-200 dark:border-[#2a2a2a] rounded-2xl shadow-2xl overflow-hidden max-h-[90vh] overflow-y-auto">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 dark:border-[#2a2a2a]">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white">
            🔐 Encrypted AI Inference
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:text-[#666] dark:hover:text-[#aaa]">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-4">

          {/* Mode selector */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-[#888] uppercase tracking-wide mb-2">
              Inference Mode
            </p>
            <div className="grid grid-cols-3 gap-2">
              {(['standard', 'concrete_ml', 'openfhe'] as const).map(m => (
                <button
                  key={m}
                  onClick={() => { setSelectedMode(m); loadModes() }}
                  className={[
                    'rounded-xl border p-3 text-left transition-all text-xs font-medium',
                    selectedMode === m
                      ? MODE_COLORS[m] + ' ring-2 ring-offset-1 ring-blue-400 dark:ring-blue-600'
                      : 'border-gray-200 dark:border-[#2a2a2a] hover:bg-gray-50 dark:hover:bg-[#1e1e1e]',
                  ].join(' ')}
                >
                  <div className="text-lg mb-1">{MODE_ICONS[m]}</div>
                  <div className="text-gray-800 dark:text-white capitalize">
                    {m.replace('_', ' ')}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Privacy steps */}
          <div className="bg-gray-50 dark:bg-[#1a1a1a] rounded-xl p-3 space-y-1">
            <p className="text-xs font-semibold text-gray-500 dark:text-[#888] uppercase tracking-wide mb-2">
              Privacy Workflow
            </p>
            {privacySteps[selectedMode].map((step, i) => (
              <p key={i} className="text-xs text-gray-600 dark:text-[#aaa]">{step}</p>
            ))}
          </div>

          {/* Upload */}
          <div
            onClick={() => fileRef.current?.click()}
            onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) handleFile(f) }}
            onDragOver={e => e.preventDefault()}
            className="border-2 border-dashed border-gray-200 dark:border-[#2a2a2a] rounded-xl p-5 text-center cursor-pointer hover:border-purple-400 dark:hover:border-purple-600 transition-all"
          >
            {preview
              ? <img src={preview} className="max-h-40 mx-auto rounded-lg object-contain" />
              : <div className="space-y-1">
                  <div className="text-3xl">🩻</div>
                  <p className="text-sm text-gray-500 dark:text-[#666]">Click or drop a medical image</p>
                  <p className="text-xs text-gray-400 dark:text-[#555]">MRI · CT scan · Dermoscopy</p>
                </div>
            }
            <input ref={fileRef} type="file" accept="image/*" className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
          </div>

          {file && <p className="text-xs text-center text-gray-400 dark:text-[#666]">{file.name}</p>}

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="w-full py-3 rounded-xl bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 dark:disabled:bg-[#333] text-white font-medium text-sm transition-all disabled:cursor-not-allowed"
          >
            {loading
              ? <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                  </svg>
                  {selectedMode === 'concrete_ml' ? 'Running FHE inference...' : 'Analyzing...'}
                </span>
              : `🔐 Analyze (${selectedMode.replace('_', ' ')})`
            }
          </button>

          {/* Error */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/10 border border-red-200 dark:border-red-900/20 rounded-xl px-4 py-3 text-sm text-red-600 dark:text-red-400">
              {error}
            </div>
          )}

          {/* Result */}
          {result && (
            <div className={`rounded-xl border p-4 space-y-3 ${
              result.error || result.status === 'not_implemented' || result.status === 'not_trained'
                ? 'border-yellow-200 dark:border-yellow-900/30 bg-yellow-50 dark:bg-yellow-900/10'
                : result.cancer_detected
                  ? 'border-red-200 dark:border-red-900/30 bg-red-50 dark:bg-red-900/10'
                  : 'border-green-200 dark:border-green-900/30 bg-green-50 dark:bg-green-900/10'
            }`}>
              {result.status === 'not_implemented' || result.status === 'not_trained'
                ? <>
                    <p className="text-sm font-semibold text-yellow-700 dark:text-yellow-300">
                      🚧 {result.inference_mode?.toUpperCase()} — {result.status === 'not_trained' ? 'Classifier Not Trained' : 'Coming Soon'}
                    </p>
                    <p className="text-xs text-yellow-600 dark:text-yellow-400">{result.message}</p>
                    {result.status === 'not_trained' && (
                      <p className="text-xs font-mono bg-yellow-100 dark:bg-yellow-900/30 rounded px-2 py-1.5 text-yellow-800 dark:text-yellow-300 mt-2 break-all">
                        venv\Scripts\python.exe -m ai.concrete_ml.fhe_classifier
                      </p>
                    )}
                  </>
                : <>
                    <div className="flex items-center gap-2">
                      <span className="text-xl">{result.cancer_detected ? '⚠️' : '✅'}</span>
                      <span className={`text-sm font-semibold ${result.cancer_detected ? 'text-red-700 dark:text-red-300' : 'text-green-700 dark:text-green-300'}`}>
                        {result.cancer_detected ? 'Cancer Signs Detected' : 'No Cancer Detected'}
                      </span>
                    </div>
                    {result.cancer_type && (
                      <p className="text-xs font-medium text-red-600 dark:text-red-400">
                        Type: {result.cancer_type}
                      </p>
                    )}
                    {result.confidence != null && (
                      <div>
                        <div className="w-full bg-gray-200 dark:bg-[#333] rounded-full h-1.5">
                          <div className={`h-1.5 rounded-full ${result.cancer_detected ? 'bg-red-500' : 'bg-green-500'}`}
                            style={{ width: `${Math.round(result.confidence * 100)}%` }} />
                        </div>
                        <p className="text-xs text-gray-500 dark:text-[#888] mt-1">
                          Confidence: {Math.round(result.confidence * 100)}%
                        </p>
                      </div>
                    )}

                    {/* Timing breakdown */}
                    <div className="grid grid-cols-3 gap-2 pt-1">
                      {[
                        { label: 'Inference', val: result.inference_time_ms },
                        { label: 'Encrypt',   val: result.encryption_time_ms },
                        { label: 'Decrypt',   val: result.decryption_time_ms },
                      ].map(({ label, val }) => (
                        <div key={label} className="bg-white dark:bg-[#1e1e1e] rounded-lg p-2 text-center border border-gray-100 dark:border-[#333]">
                          <p className="text-xs text-gray-400 dark:text-[#666]">{label}</p>
                          <p className="text-xs font-bold text-gray-700 dark:text-[#ccc]">
                            {val != null && val >= 0 ? `${val}ms` : '—'}
                          </p>
                        </div>
                      ))}
                    </div>

                    <div className="flex items-center gap-2 pt-1">
                      <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-[#242424] text-gray-600 dark:text-[#aaa]">
                        {MODE_ICONS[result.inference_mode]} {result.inference_mode?.replace('_', ' ')}
                      </span>
                      {result.security_level_bits > 0 && (
                        <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300">
                          🔒 {result.security_level_bits}-bit
                        </span>
                      )}
                    </div>
                  </>
              }

              <div className="bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-100 dark:border-yellow-900/20 rounded-lg px-3 py-2">
                <p className="text-xs text-yellow-800 dark:text-yellow-200 font-medium">
                  ⚕️ This result is based on trained dataset. Consult doctor.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
