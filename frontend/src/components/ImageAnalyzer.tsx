import { useState, useRef } from 'react'
import ReactMarkdown from 'react-markdown'

// Use current page origin so it works on any IP/device without rebuilding
const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface AnalysisResult {
  cancer_detected: boolean
  cancer_type: string | null
  confidence: number
  message: string
  match_source?: string
  safety_message?: string
  unknown_image?: boolean
  low_confidence?: boolean
}

interface ImageAnalyzerProps {
  token: string
  darkMode: boolean
  onClose: () => void
}

export default function ImageAnalyzer({ token, darkMode, onClose }: ImageAnalyzerProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    setResult(null)
    setError('')
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (!file || !file.type.startsWith('image/')) return
    setSelectedFile(file)
    setResult(null)
    setError('')
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch(`${API_URL}/images/analyze`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Analysis failed')
      }

      const data: AnalysisResult = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze image')
    } finally {
      setLoading(false)
    }
  }

  const confidencePct = result ? Math.round(result.confidence * 100) : 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-all duration-300">
      <style>{`
        .result-markdown p  { margin: 0.3em 0; }
        .result-markdown ul, .result-markdown ol { padding-left: 1.4em; margin: 0.3em 0; }
        .result-markdown li { margin: 0.2em 0; }
        .result-markdown strong { font-weight: 700; color: inherit; }
        .result-markdown h1, .result-markdown h2, .result-markdown h3 { font-weight: 700; margin: 0.6em 0 0.3em; color: inherit; }
        .dark .result-markdown strong { color: #fff; }
        .dark .result-markdown h1, .dark .result-markdown h2, .dark .result-markdown h3 { color: #fff; }
        .dark .result-markdown p, .dark .result-markdown li { color: #fecaca; }
      `}</style>
      <div className="w-full max-w-md bg-white dark:bg-[#161616] border border-gray-200 dark:border-[#2a2a2a] rounded-2xl shadow-2xl overflow-hidden transition-all duration-300">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 dark:border-[#2a2a2a]">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white">
            🔬 Medical Image Analyzer
          </h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 dark:text-[#666] hover:bg-gray-100 dark:hover:bg-[#2a2a2a] hover:text-gray-600 dark:hover:text-[#aaa] transition-all"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-5 space-y-4 max-h-[80vh] overflow-y-auto">
          {/* Upload Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
            className="relative border-2 border-dashed border-gray-200 dark:border-[#2a2a2a] rounded-xl p-6 text-center cursor-pointer hover:border-blue-500 dark:hover:border-blue-500 hover:bg-gray-50 dark:hover:bg-[#1e1e1e] transition-all group"
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Preview"
                className="max-h-48 mx-auto rounded-lg object-contain"
              />
            ) : (
              <div className="space-y-2">
                <svg className="w-10 h-10 mx-auto text-gray-400 dark:text-[#555] group-hover:text-blue-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-sm font-medium text-gray-700 dark:text-[#ccc]">
                  Click or drag to upload medical image
                </p>
                <p className="text-xs text-gray-400 dark:text-[#666]">
                  X-ray, MRI, CT scan, skin, pathology images
                </p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          {selectedFile && (
            <p className="text-xs text-center text-gray-400 dark:text-[#666]">
              {selectedFile.name}
            </p>
          )}

          {/* Analyze Button */}
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className="w-full py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white font-medium rounded-xl transition-all text-sm disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing...
              </span>
            ) : 'Analyze Image'}
          </button>

          {/* Error */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/10 border border-red-100 dark:border-red-900/20 text-red-600 dark:text-red-400 px-4 py-3 rounded-xl text-sm">
              {error}
            </div>
          )}

          {/* Result */}
          {result && (() => {
            // Unknown image — same card style, different content
            if (result.unknown_image) {
              return (
                <div className="rounded-xl border border-gray-200 dark:border-[#2a2a2a] bg-gray-50 dark:bg-[#1a1a1a] p-4 space-y-3 transition-colors duration-300">
                  {/* Badge */}
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">❓</span>
                    <span className="text-base font-semibold text-gray-600 dark:text-gray-300">
                      Unknown Image
                    </span>
                  </div>

                  {/* Message */}
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    This image type was not included in the training dataset.
                  </p>

                  {/* Safety Disclaimer */}
                  <div className="bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-100 dark:border-yellow-900/20 rounded-lg px-3 py-2">
                    <p className="text-xs text-yellow-800 dark:text-yellow-200 font-medium">
                      ⚕️ {result.safety_message || 'This result is AI prediction only. Consult a doctor.'}
                    </p>
                  </div>
                </div>
              )
            }

            // Low confidence — same card style, yellow tones
            if (result.low_confidence) {
              return (
                <div className="rounded-xl border border-yellow-200 dark:border-yellow-900/30 bg-yellow-50 dark:bg-yellow-900/10 p-4 space-y-3 transition-colors duration-300">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">⚠️</span>
                    <span className="text-base font-semibold text-yellow-700 dark:text-yellow-300">
                      Low Confidence
                    </span>
                  </div>
                  <p className="text-sm text-yellow-700 dark:text-yellow-400">
                    The model is not confident enough to make a reliable prediction.
                  </p>
                  <div className="bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-100 dark:border-yellow-900/20 rounded-lg px-3 py-2">
                    <p className="text-xs text-yellow-800 dark:text-yellow-200 font-medium">
                      ⚕️ {result.safety_message || 'This result is AI prediction only. Consult a doctor.'}
                    </p>
                  </div>
                </div>
              )
            }

            // Normal result — cancer or no cancer
            return (
              <div className={`rounded-xl border p-4 space-y-3 transition-colors duration-300 ${result.cancer_detected
                  ? 'border-red-200 dark:border-red-900/30 bg-red-50 dark:bg-red-900/10'
                  : 'border-green-200 dark:border-green-900/30 bg-green-50 dark:bg-green-900/10'
                }`}>
                {/* Detection Badge */}
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{result.cancer_detected ? '⚠️' : '✅'}</span>
                  <span className={`text-base font-semibold ${result.cancer_detected ? 'text-red-700' : 'text-green-700'}`}>
                    {result.cancer_detected ? 'Cancer Signs Detected' : 'No Cancer Detected'}
                  </span>
                </div>

                {/* Cancer Type */}
                {result.cancer_detected && result.cancer_type && (
                  <div>
                    <p className="text-xs text-red-600 font-medium uppercase tracking-wide">Cancer Type</p>
                    <p className="text-sm font-semibold text-red-800">{result.cancer_type}</p>
                  </div>
                )}

                {/* Confidence Bar */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className={result.cancer_detected ? 'text-red-600' : 'text-green-600'}>
                      {result.cancer_detected ? 'Risk Score' : 'Confidence (No Cancer)'}
                    </span>
                    <span className={`font-semibold ${result.cancer_detected ? 'text-red-700' : 'text-green-700'}`}>
                      {confidencePct}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${result.cancer_detected ? 'bg-red-500' : 'bg-green-500'}`}
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>
                </div>

                {/* Message */}
                <div className={`text-sm max-w-none transition-colors duration-300 result-markdown ${result.cancer_detected ? 'text-red-800 dark:text-red-100' : 'text-green-800 dark:text-green-100'}`}>
                  <ReactMarkdown>{result.message}</ReactMarkdown>
                </div>

                {/* Match source */}
                {result.match_source && result.match_source !== 'none' && (
                  <p className="text-xs text-gray-500">
                    Matched from: {result.match_source === 'skin_csv' ? 'Skin Cancer CSV Dataset' : result.match_source === 'brain' ? 'Brain Cancer Dataset' : 'Lung Cancer Dataset'}
                  </p>
                )}

                {/* Safety Disclaimer */}
                <div className="bg-yellow-50 dark:bg-yellow-900/10 border border-yellow-100 dark:border-yellow-900/20 rounded-lg px-3 py-2">
                  <p className="text-xs text-yellow-800 dark:text-yellow-200 font-medium">
                    ⚕️ {result.safety_message || 'This result is AI prediction only. Consult a doctor.'}
                  </p>
                </div>
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
