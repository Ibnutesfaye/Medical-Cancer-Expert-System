import { useState, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface Patient {
  user_id: number
  username: string
  scan_count: number
  last_scan: string | null
  positive_scans: number
}

interface Analysis {
  id: number
  filename: string
  cancer_detected: boolean
  cancer_type: string | null
  confidence: number | null
  model_used: string
  created_at: string
}

interface Timeline {
  patient_id: number
  total_scans: number
  timeline: Analysis[]
  trend: 'increasing' | 'decreasing' | 'stable'
}

interface DoctorNote {
  id: number
  note_type: string
  content: string
  is_private: boolean
  created_at: string
}

interface DashboardStats {
  total_analyses: number
  cancer_detected: number
  healthy: number
  total_patients: number
  total_notes: number
  detection_rate: number
}

interface DoctorDashboardProps {
  token: string
  onBack: () => void
}

const TREND_ICONS: Record<string, string> = {
  increasing: '📈',
  decreasing: '📉',
  stable:     '➡️',
}

const NOTE_TYPES = ['observation', 'diagnosis', 'recommendation', 'followup']

export default function DoctorDashboard({ token, onBack }: DoctorDashboardProps) {
  const [stats, setStats]           = useState<DashboardStats | null>(null)
  const [patients, setPatients]     = useState<Patient[]>([])
  const [selected, setSelected]     = useState<Patient | null>(null)
  const [timeline, setTimeline]     = useState<Timeline | null>(null)
  const [notes, setNotes]           = useState<DoctorNote[]>([])
  const [newNote, setNewNote]       = useState('')
  const [noteType, setNoteType]     = useState('observation')
  const [savingNote, setSavingNote] = useState(false)
  const [gradcamId, setGradcamId]   = useState<number | null>(null)
  const [gradcamB64, setGradcamB64] = useState<string | null>(null)
  const [gradcamLoading, setGradcamLoading] = useState(false)
  const [activeTab, setActiveTab]   = useState<'overview' | 'patients' | 'notes'>('overview')
  const [loading, setLoading]       = useState(false)

  useEffect(() => { fetchStats(); fetchPatients() }, [])

  const h = { Authorization: `Bearer ${token}` }

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_URL}/doctor/dashboard/stats`, { headers: h })
      if (res.ok) setStats(await res.json())
    } catch { /* ignore */ }
  }

  const fetchPatients = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/doctor/patients`, { headers: h })
      if (res.ok) setPatients(await res.json())
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }

  const selectPatient = async (p: Patient) => {
    setSelected(p)
    setTimeline(null)
    setNotes([])
    setGradcamB64(null)
    setActiveTab('patients')
    try {
      const [tRes, nRes] = await Promise.all([
        fetch(`${API_URL}/doctor/patients/${p.user_id}/analyses`, { headers: h }),
        fetch(`${API_URL}/doctor/notes/${p.user_id}`, { headers: h }),
      ])
      if (tRes.ok) setTimeline(await tRes.json())
      if (nRes.ok) setNotes(await nRes.json())
    } catch { /* ignore */ }
  }

  const saveNote = async () => {
    if (!selected || !newNote.trim()) return
    setSavingNote(true)
    try {
      const res = await fetch(`${API_URL}/doctor/notes`, {
        method: 'POST',
        headers: { ...h, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: selected.user_id,
          note_type: noteType,
          content: newNote.trim(),
        }),
      })
      if (res.ok) {
        setNewNote('')
        const nRes = await fetch(`${API_URL}/doctor/notes/${selected.user_id}`, { headers: h })
        if (nRes.ok) setNotes(await nRes.json())
      }
    } catch { /* ignore */ }
    finally { setSavingNote(false) }
  }

  const viewGradCAM = async (analysisId: number) => {
    setGradcamId(analysisId)
    setGradcamB64(null)
    setGradcamLoading(true)
    // GradCAM requires re-uploading the image — show placeholder info instead
    setGradcamLoading(false)
    setGradcamB64('placeholder')
  }

  const formatDate = (d: string) => new Date(d).toLocaleDateString()

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-gray-50 dark:bg-[#0f0f0f] text-gray-900 dark:text-white">

      {/* Header */}
      <header className="bg-white dark:bg-[#161616] border-b border-gray-200 dark:border-[#2a2a2a] px-6 py-4 flex items-center gap-4 flex-shrink-0">
        <button onClick={onBack} className="text-gray-400 hover:text-gray-600 dark:text-[#666] dark:hover:text-[#aaa]">
          <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <h1 className="text-lg font-bold">👨‍⚕️ Doctor Dashboard</h1>
      </header>

      {/* Tabs */}
      <div className="bg-white dark:bg-[#161616] border-b border-gray-100 dark:border-[#2a2a2a] px-6">
        <div className="flex gap-6">
          {(['overview', 'patients', 'notes'] as const).map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`py-3 text-sm font-medium border-b-2 transition-colors capitalize ${
                activeTab === tab
                  ? 'border-blue-600 dark:border-blue-500 text-gray-900 dark:text-white'
                  : 'border-transparent text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#aaa]'
              }`}>
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto px-6 py-6">

          {/* Overview tab */}
          {activeTab === 'overview' && stats && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {[
                  { label: 'Total Analyses',  val: stats.total_analyses,  color: 'text-blue-600' },
                  { label: 'Cancer Detected', val: stats.cancer_detected,  color: 'text-red-600' },
                  { label: 'Healthy',          val: stats.healthy,          color: 'text-green-600' },
                  { label: 'Total Patients',   val: stats.total_patients,   color: 'text-purple-600' },
                  { label: 'Clinical Notes',   val: stats.total_notes,      color: 'text-yellow-600' },
                  { label: 'Detection Rate',   val: `${stats.detection_rate}%`, color: 'text-orange-600' },
                ].map(({ label, val, color }) => (
                  <div key={label} className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5 shadow-sm">
                    <p className="text-xs text-gray-400 dark:text-[#666] mb-1">{label}</p>
                    <p className={`text-2xl font-bold ${color}`}>{val}</p>
                  </div>
                ))}
              </div>

              <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5">
                <h3 className="text-sm font-semibold mb-4">Patient List</h3>
                {loading ? <p className="text-sm text-gray-400 dark:text-[#666]">Loading...</p> : (
                  <div className="space-y-2">
                    {patients.slice(0, 5).map(p => (
                      <div key={p.user_id}
                        onClick={() => selectPatient(p)}
                        className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-[#1a1a1a] cursor-pointer transition-colors border border-gray-50 dark:border-[#2a2a2a]">
                        <div>
                          <p className="text-sm font-medium">{p.username}</p>
                          <p className="text-xs text-gray-400 dark:text-[#666]">
                            {p.scan_count} scans · Last: {p.last_scan ? formatDate(p.last_scan) : 'N/A'}
                          </p>
                        </div>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          p.positive_scans > 0
                            ? 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400'
                            : 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400'
                        }`}>
                          {p.positive_scans > 0 ? `${p.positive_scans} positive` : 'Healthy'}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Patients tab */}
          {activeTab === 'patients' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              {/* Patient list sidebar */}
              <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-4 space-y-2 h-fit">
                <h3 className="text-xs font-semibold text-gray-500 dark:text-[#888] uppercase tracking-wide mb-3">Patients</h3>
                {patients.map(p => (
                  <button key={p.user_id} onClick={() => selectPatient(p)}
                    className={`w-full text-left p-3 rounded-lg transition-colors text-sm ${
                      selected?.user_id === p.user_id
                        ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-900/40'
                        : 'hover:bg-gray-50 dark:hover:bg-[#1a1a1a] border border-transparent'
                    }`}>
                    <p className="font-medium">{p.username}</p>
                    <p className="text-xs text-gray-400 dark:text-[#666]">{p.scan_count} scans</p>
                  </button>
                ))}
              </div>

              {/* Timeline */}
              <div className="md:col-span-2 space-y-4">
                {selected && timeline ? (
                  <>
                    <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold">{selected.username}</h3>
                        <span className="text-sm text-gray-500 dark:text-[#888]">
                          {TREND_ICONS[timeline.trend]} Trend: {timeline.trend}
                        </span>
                      </div>

                      {/* Confidence chart (simple bars) */}
                      <div className="space-y-2">
                        {timeline.timeline.map((a, i) => (
                          <div key={a.id} className="flex items-center gap-3">
                            <span className="text-xs text-gray-400 dark:text-[#666] w-20 flex-shrink-0">
                              {formatDate(a.created_at)}
                            </span>
                            <div className="flex-1 bg-gray-100 dark:bg-[#333] rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${a.cancer_detected ? 'bg-red-500' : 'bg-green-500'}`}
                                style={{ width: `${Math.round((a.confidence ?? 0) * 100)}%` }}
                              />
                            </div>
                            <span className="text-xs font-medium w-12 text-right">
                              {a.confidence != null ? `${Math.round(a.confidence * 100)}%` : '—'}
                            </span>
                            <span className={`text-xs px-2 py-0.5 rounded-full ${
                              a.cancer_detected
                                ? 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400'
                                : 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400'
                            }`}>
                              {a.cancer_type || (a.cancer_detected ? 'Cancer' : 'Healthy')}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-8 text-center text-gray-400 dark:text-[#666]">
                    Select a patient to view their scan history
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Notes tab */}
          {activeTab === 'notes' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Add note */}
              <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5 space-y-4">
                <h3 className="text-sm font-semibold">Add Clinical Note</h3>
                {!selected ? (
                  <p className="text-sm text-gray-400 dark:text-[#666]">
                    Select a patient from the Patients tab first
                  </p>
                ) : (
                  <>
                    <p className="text-xs text-blue-600 dark:text-blue-400 font-medium">
                      Patient: {selected.username}
                    </p>
                    <select
                      value={noteType}
                      onChange={e => setNoteType(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-[#2a2a2a] bg-white dark:bg-[#1a1a1a] text-sm text-gray-700 dark:text-[#ccc]"
                    >
                      {NOTE_TYPES.map(t => (
                        <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
                      ))}
                    </select>
                    <textarea
                      rows={5}
                      value={newNote}
                      onChange={e => setNewNote(e.target.value)}
                      placeholder="Enter clinical note..."
                      className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-[#2a2a2a] bg-white dark:bg-[#1a1a1a] text-sm text-gray-700 dark:text-[#ccc] resize-none focus:outline-none focus:border-blue-400"
                    />
                    <button
                      onClick={saveNote}
                      disabled={savingNote || !newNote.trim()}
                      className="w-full py-2 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200 dark:disabled:bg-[#333] text-white text-sm font-medium transition-all"
                    >
                      {savingNote ? 'Saving...' : '💾 Save Note'}
                    </button>
                  </>
                )}
              </div>

              {/* Notes list */}
              <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-5">
                <h3 className="text-sm font-semibold mb-4">
                  {selected ? `Notes for ${selected.username}` : 'Clinical Notes'}
                </h3>
                {notes.length === 0
                  ? <p className="text-sm text-gray-400 dark:text-[#666]">No notes yet</p>
                  : <div className="space-y-3">
                      {notes.map(n => (
                        <div key={n.id} className="p-3 rounded-lg bg-gray-50 dark:bg-[#1a1a1a] border border-gray-100 dark:border-[#2a2a2a]">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-blue-600 dark:text-blue-400 capitalize">
                              {n.note_type}
                            </span>
                            <span className="text-xs text-gray-400 dark:text-[#666]">
                              {formatDate(n.created_at)}
                            </span>
                          </div>
                          <p className="text-sm text-gray-700 dark:text-[#ccc]">{n.content}</p>
                        </div>
                      ))}
                    </div>
                }
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
