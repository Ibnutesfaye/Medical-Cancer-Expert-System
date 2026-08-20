import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import ImageAnalyzer from './ImageAnalyzer'
import EncryptedAI from './EncryptedAI'
import Sidebar from './Sidebar'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  source?: string
  timestamp: Date
}

interface Citation {
  document_name?: string
  page_number?: number
  chunk_text: string
  relevance_score: number
  source_type?: string
  url?: string
}

interface ChatSession {
  id: string
  title: string
  date: string
  messages: Message[]
}

interface ChatInterfaceProps {
  token: string
  username?: string
  onLogout?: () => void
  onAdminClick?: () => void
  onBenchmarkClick?: () => void
  onDoctorClick?: () => void
  onResearchClick?: () => void
  onSettings?: () => void
  onHelp?: () => void
}

function getDateLabel(d: Date): string {
  const now = new Date()
  const diff = (now.getTime() - d.getTime()) / (1000 * 60 * 60 * 24)
  if (diff < 1) return 'Today'
  if (diff < 2) return 'Yesterday'
  if (diff < 30) return '30 Days'
  return d.toISOString().slice(0, 7)
}

const SUGGESTIONS = [
  { icon: '💊', label: 'Treatment options', query: 'What are the treatment options for cancer?' },
  { icon: '🩺', label: 'Symptoms & diagnosis', query: 'What are the symptoms and diagnosis methods for cancer?' },
  { icon: '🛡️', label: 'Prevention methods', query: 'What are the prevention methods for cancer?' },
  { icon: '📚', label: 'Cancer types', query: 'What are the different types of cancer?' },
]

export default function ChatInterface({ token, username = "User", onLogout, onAdminClick, onBenchmarkClick, onDoctorClick, onResearchClick, onSettings, onHelp }: ChatInterfaceProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showImageAnalyzer, setShowImg] = useState(false)
  const [showEncryptedAI, setShowEncryptedAI] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 1024)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const recognitionRef = useRef<any>(null)
  const synthRef = useRef<SpeechSynthesisUtterance | null>(null)

  // On desktop (≥1024px) sidebar is always visible via CSS; we only manage the drawer state
  useEffect(() => {
    const saved = localStorage.getItem('chat_sessions')
    if (!saved) return
    try {
      const parsed: ChatSession[] = JSON.parse(saved)
      const hydrated = parsed.map(s => ({
        ...s,
        messages: s.messages.map(m => ({ ...m, timestamp: new Date(m.timestamp) })),
      }))
      setSessions(hydrated)
      if (hydrated.length > 0) {
        setActiveId(hydrated[0].id)
        setMessages(hydrated[0].messages)
      }
    } catch { }
  }, [])

  useEffect(() => {
    if (sessions.length > 0) localStorage.setItem('chat_sessions', JSON.stringify(sessions))
  }, [sessions])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
  }, [messages])

  useEffect(() => () => {
    window.speechSynthesis?.cancel()
    recognitionRef.current?.stop()
  }, [])

  // ── session helpers ──────────────────────────────────────────────────────────
  const newChat = () => { setMessages([]); setActiveId(null); setInput(''); setError('') }

  const selectSession = (id: string) => {
    const s = sessions.find(s => s.id === id)
    if (s) { setMessages(s.messages); setActiveId(id) }
  }

  const deleteSession = (id: string) => {
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id)
      if (next.length === 0) localStorage.removeItem('chat_sessions')
      else localStorage.setItem('chat_sessions', JSON.stringify(next))
      return next
    })
    if (activeSessionId === id) newChat()
  }

  const saveSession = (msgs: Message[]) => {
    if (!msgs.length) return
    const title = msgs[0].content.slice(0, 48) + (msgs[0].content.length > 48 ? '…' : '')
    const date = getDateLabel(msgs[0].timestamp)
    setSessions(prev => {
      if (activeSessionId) return prev.map(s => s.id === activeSessionId ? { ...s, messages: msgs, title } : s)
      const id = Date.now().toString()
      setActiveId(id)
      return [{ id, title, date, messages: msgs }, ...prev]
    })
  }

  // ── send ─────────────────────────────────────────────────────────────────────
  const handleSend = async (overrideQuery?: string) => {
    const query = (overrideQuery ?? input).trim()
    if (!query) return
    setError(''); setInput(''); setLoading(true)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: query, timestamp: new Date() }
    const next = [...messages, userMsg]
    setMessages(next)

    try {
      const history = messages.slice(-6).map(m => ({ role: m.role, content: m.content }))
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ query, conversation_history: history }),
      })
      if (!res.ok) {
        if (res.status === 401) { localStorage.removeItem('auth_token'); onLogout?.(); return }
        throw new Error('Failed to get response')
      }

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let content = ''
      let citations: Citation[] = []
      const aMsg: Message = { id: (Date.now() + 1).toString(), role: 'assistant', content: '', citations: [], source: 'document', timestamp: new Date() }
      setMessages([...next, aMsg])

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          for (const line of decoder.decode(value).split('\n')) {
            if (!line.startsWith('data: ')) continue
            const data = line.slice(6)
            if (data === '[DONE]') break
            else if (data.startsWith('[CITATIONS]')) { try { citations = JSON.parse(data.slice(11)) } catch { } }
            else if (data.startsWith('[SOURCE]')) {
              const src = data.slice(8)
              setMessages(prev => { const u = [...prev]; u[u.length - 1] = { ...u[u.length - 1], source: src, citations }; saveSession(u); return u })
            } else if (data.startsWith('[ERROR]')) { throw new Error(data.slice(7)) }
            else {
              content += data.replace(/\\n/g, '\n')
              setMessages(prev => { const u = [...prev]; u[u.length - 1] = { ...u[u.length - 1], content, citations }; return u })
            }
          }
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to send message')
    } finally {
      setLoading(false)
    }
  }

  // ── voice ────────────────────────────────────────────────────────────────────
  const handleVoice = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) { setError('Voice input not supported in this browser.'); return }
    if (isRecording) { recognitionRef.current?.stop(); setIsRecording(false); return }
    const r = new SR(); r.lang = 'en-US'; r.interimResults = false
    r.onstart = () => setIsRecording(true)
    r.onresult = (e: any) => setInput(p => p ? p + ' ' + e.results[0][0].transcript : e.results[0][0].transcript)
    r.onerror = () => setIsRecording(false)
    r.onend = () => setIsRecording(false)
    recognitionRef.current = r; r.start()
  }

  const handleTTS = (text: string, id: string) => {
    if (!('speechSynthesis' in window)) return
    if (playingId === id) { window.speechSynthesis.cancel(); setPlayingId(null); return }
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text); u.rate = 0.9
    u.onstart = () => setPlayingId(id)
    u.onend = () => setPlayingId(null)
    u.onerror = () => setPlayingId(null)
    synthRef.current = u; window.speechSynthesis.speak(u)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const handleTextarea = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    const el = e.target; el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 140) + 'px'
  }

  const chatHistory = sessions.map(s => ({ id: s.id, title: s.title, date: s.date }))

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-50 dark:bg-[#0f0f0f] transition-colors duration-300" style={{ fontFamily: 'Inter, sans-serif' }}>

      {/* Image Analyzer Modal */}
      {showImageAnalyzer && <ImageAnalyzer token={token} darkMode onClose={() => setShowImg(false)} />}

      {/* Encrypted AI Modal */}
      {showEncryptedAI && <EncryptedAI token={token} onClose={() => setShowEncryptedAI(false)} />}

      {/* ── SIDEBAR ── */}
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={newChat}
        chatHistory={chatHistory}
        activeId={activeSessionId}
        onSelectChat={selectSession}
        onDeleteChat={deleteSession}
        username={username}
        onAdminClick={onAdminClick}
        onBenchmarkClick={onBenchmarkClick}
        onDoctorClick={onDoctorClick}
        onResearchClick={onResearchClick}
        onLogout={onLogout}
        onSettings={onSettings}
        onHelp={onHelp}
      />

      {/* ── MAIN ── */}
      <div className="flex flex-col flex-1 min-w-0 h-full overflow-hidden">

        {/* Top bar */}
        <header className="flex items-center gap-3 px-4 h-[52px] border-b border-gray-200 dark:border-[#1e1e1e] bg-white dark:bg-[#0f0f0f] flex-shrink-0 transition-colors duration-300">
          {/* Hamburger — always visible, sidebar handles its own desktop visibility */}
          <button
            onClick={() => setSidebarOpen(v => !v)}
            className="text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#aaa] transition-colors p-1.5 rounded-lg"
            aria-label="Toggle sidebar"
          >
            <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="text-sm font-semibold text-gray-400 dark:text-[#777]">Medical Cancer Expert System</span>
        </header>

        {/* Messages scroll area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-[720px] w-full mx-auto px-4 sm:px-6 py-8">

            {/* ── WELCOME ── */}
            {messages.length === 0 && (
              <div className="flex flex-col items-center text-center pt-12 sm:pt-20">

                {/* Gradient circle */}
                <div className="relative flex items-center justify-center mb-8"
                  style={{ width: 'min(60vw, 260px)', height: 'min(60vw, 260px)' }}>
                  <div
                    className="absolute rounded-full opacity-20 blur-3xl"
                    style={{
                      width: '100%', height: '100%',
                      background: 'radial-gradient(circle, #a855f7 0%, #ec4899 60%, transparent 100%)',
                    }}
                  />
                  <img src="/insa-logo.webp" alt="Logo" className="relative w-16 h-16 sm:w-20 sm:h-20 object-contain shadow-2xl rounded-2xl" />
                </div>

                <h1 className="text-xl sm:text-3xl lg:text-4xl font-bold text-gray-900 dark:text-white tracking-tight mb-3 px-2">
                  Welcome to Medical Cancer Expert System
                </h1>
                <p className="text-sm sm:text-base text-gray-500 dark:text-[#666] mb-10 max-w-md px-2">
                  Ask me anything about cancer from medical documents
                </p>

                {/* Suggestion cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-[520px]">
                  {SUGGESTIONS.map(({ icon, label, query }) => (
                    <button
                      key={label}
                      onClick={() => handleSend(query)}
                      className="flex items-center gap-3 px-4 py-4 rounded-2xl bg-white dark:bg-[#161616] border border-gray-200 dark:border-[#2a2a2a] text-gray-700 dark:text-[#ccc] text-sm font-medium text-left hover:bg-gray-50 dark:hover:bg-[#1e1e1e] hover:border-gray-300 dark:hover:border-[#3a3a3a] transition-all shadow-sm"
                    >
                      <span className="text-xl flex-shrink-0">{icon}</span>
                      <span>{label}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* ── MESSAGES ── */}
            {messages.map(msg => (
              <div key={msg.id} className="mb-7">
                {msg.role === 'user' ? (
                  <div className="flex justify-end">
                      <div className="max-w-[85%] sm:max-w-[72%] px-4 py-3 rounded-2xl bg-blue-600 dark:bg-[#1e3a5f] text-white dark:text-[#e8f0fe] text-sm leading-relaxed shadow-lg break-words">
                        {msg.content}
                      </div>
                  </div>
                ) : (
                  <div className="flex gap-3 sm:gap-4 items-start">
                    <img src="/insa-logo.webp" alt="Logo" className="w-8 h-8 rounded-xl flex-shrink-0 mt-0.5 object-contain" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm leading-[1.75] text-gray-700 dark:text-[#d4d4d4] md-content break-words">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>

                      {/* Source + TTS */}
                      {msg.source && msg.content && (
                        <div className="flex flex-wrap gap-2 mt-3 items-center">
                          {msg.source === 'document' && <Badge bg="#1a3a2a" color="#4ade80">📄 Documents</Badge>}
                          {msg.source === 'wikipedia' && <Badge bg="#1a2a3a" color="#60a5fa">🌐 Wikipedia</Badge>}
                          {msg.source === 'pubmed' && <Badge bg="#2a1a3a" color="#c084fc">🔬 PubMed</Badge>}
                          {(msg.source === 'llm' || msg.source === 'external') && <Badge bg="#3a2a1a" color="#fb923c">🤖 AI Knowledge</Badge>}
                          <button
                            onClick={() => handleTTS(msg.content, msg.id)}
                            className={[
                              'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border transition-all',
                              playingId === msg.id
                                ? 'bg-blue-600 border-blue-600 text-white'
                                : 'bg-transparent border-gray-200 dark:border-[#333] text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#aaa] hover:border-gray-300 dark:hover:border-[#555]',
                            ].join(' ')}
                          >
                            {playingId === msg.id ? '⏹ Stop' : '🔊 Listen'}
                          </button>
                        </div>
                      )}

                      {/* Citations */}
                      {msg.citations && msg.citations.length > 0 && (
                        <div className="mt-3 p-3 bg-gray-50 dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a]">
                          <p className="text-[11px] font-semibold text-gray-400 dark:text-[#555] uppercase tracking-wider mb-1.5">Sources</p>
                          {msg.citations.map((c, i) => (
                            <div key={i} className="text-xs text-gray-500 dark:text-[#666] mb-1">
                              {c.source_type === 'document' || !c.url
                                ? <span>[{i + 1}] {c.document_name} — p.{c.page_number}</span>
                                : <a href={c.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-[#4f8ef7] hover:underline">
                                  [{i + 1}] {c.source_type === 'wikipedia' ? '🌐 Wikipedia' : '🔬 PubMed'} — {c.url}
                                </a>
                              }
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Typing dots */}
            {loading && (
              <div className="flex gap-3 sm:gap-4 items-start mb-7">
                <img src="/insa-logo.webp" alt="Logo" className="w-8 h-8 rounded-xl flex-shrink-0 object-contain" />
                <div className="flex gap-1.5 items-center pt-2.5">
                  {[0, 150, 300].map((delay, i) => (
                    <span key={i} className="w-2 h-2 rounded-full bg-[#444] animate-bounce" style={{ animationDelay: `${delay}ms` }} />
                  ))}
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── INPUT AREA ── */}
        <div className="flex-shrink-0 bg-white dark:bg-[#0f0f0f] border-t border-gray-100 dark:border-[#1e1e1e] px-3 sm:px-6 pt-3 pb-4 sm:pb-5 transition-colors duration-300">
          <div className="max-w-[720px] w-full mx-auto">

            {error && (
              <div className="mb-2.5 px-4 py-2.5 rounded-xl bg-red-50 dark:bg-[#2a1a1a] border border-red-100 dark:border-[#5a2a2a] text-red-600 dark:text-red-400 text-sm">
                {error}
              </div>
            )}

            {/* Input box */}
            <div className="bg-gray-50 dark:bg-[#161616] border border-gray-200 dark:border-[#2a2a2a] rounded-2xl px-3 py-3 shadow-sm dark:shadow-2xl focus-within:border-gray-300 dark:focus-within:border-[#3a3a3a] transition-all">
              <div className="flex items-end gap-2">

                {/* Attachment — Standard image analysis */}
                <button
                  onClick={() => setShowImg(true)}
                  title="Analyze medical image"
                  className="text-[#555] hover:text-[#888] transition-colors p-1 rounded-lg flex-shrink-0"
                >
                  <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                  </svg>
                </button>

                {/* Encrypted AI inference */}
                <button
                  onClick={() => setShowEncryptedAI(true)}
                  title="Encrypted AI (FHE) inference"
                  className="text-[#555] hover:text-purple-500 transition-colors p-1 rounded-lg flex-shrink-0 text-base leading-none"
                >
                  🔐
                </button>

                {/* Textarea */}
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={handleTextarea}
                  onKeyDown={handleKeyDown}
                  placeholder="Message MedAssist"
                  rows={1}
                  disabled={loading}
                  className="flex-1 min-w-0 bg-transparent border-none outline-none text-gray-900 dark:text-[#e0e0e0] text-sm leading-relaxed resize-none placeholder-gray-400 dark:placeholder-[#444]"
                  style={{ maxHeight: 140, overflowY: 'auto', caretColor: '#4f8ef7', fontFamily: 'Inter, sans-serif' }}
                />

                {/* Mic */}
                <button
                  onClick={handleVoice}
                  title="Voice input"
                  className={[
                    'flex-shrink-0 p-1 rounded-lg transition-all',
                    isRecording ? 'text-white bg-red-500' : 'text-[#555] hover:text-[#888]',
                  ].join(' ')}
                >
                  <svg width="18" height="18" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                  </svg>
                </button>

                {/* Send */}
                <button
                  onClick={() => handleSend()}
                  disabled={loading || !input.trim()}
                  className={[
                    'flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center transition-all',
                    loading || !input.trim()
                      ? 'bg-[#222] text-[#444] cursor-not-allowed'
                      : 'bg-[#1a56db] text-white hover:bg-[#1e4fc2] shadow-[0_2px_8px_rgba(26,86,219,0.4)]',
                  ].join(' ')}
                >
                  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Pill buttons */}
            <div className="flex flex-wrap justify-center gap-2 mt-3">
              <button
                onClick={() => handleSend(input ? `Think deeply and analyze: ${input}` : undefined)}
                className="px-4 py-1.5 rounded-full text-xs font-medium bg-transparent border border-gray-200 dark:border-[#2a2a2a] text-gray-500 dark:text-[#666] hover:border-gray-300 dark:hover:border-[#444] hover:text-gray-700 dark:hover:text-[#aaa] transition-all"
              >
                🧠 DeepThink
              </button>
              <button
                onClick={() => handleSend(input ? `Search and find information about: ${input}` : undefined)}
                className="px-4 py-1.5 rounded-full text-xs font-medium bg-transparent border border-gray-200 dark:border-[#2a2a2a] text-gray-500 dark:text-[#666] hover:border-gray-300 dark:hover:border-[#444] hover:text-gray-700 dark:hover:text-[#aaa] transition-all"
              >
                🔍 Search
              </button>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 4px; }
        .dark ::-webkit-scrollbar-thumb { background: #2a2a2a; }
        .md-content p  { margin: 0.35em 0; }
        .md-content ul, .md-content ol { padding-left: 1.4em; margin: 0.35em 0; }
        .md-content li { margin: 0.2em 0; }
        .md-content code { font-size: 12px; padding: 2px 6px; border-radius: 5px; background: #f3f4f6; color: #0891b2; font-family: monospace; }
        .dark .md-content code { background: #1e1e1e; color: #a5f3fc; }
        .md-content pre { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; overflow-x: auto; margin: 8px 0; }
        .dark .md-content pre { background: #1a1a1a; border: 1px solid #2a2a2a; }
        .md-content pre code { background: none; padding: 0; }
        .md-content strong { color: inherit; font-weight: 700; }
        .dark .md-content strong { color: #fff; }
        .md-content h1, .md-content h2, .md-content h3 { color: #111827; margin: 0.8em 0 0.4em; }
        .dark .md-content h1, .dark .md-content h2, .dark .md-content h3 { color: #fff; }
        .md-content a { color: #2563eb; }
        .dark .md-content a { color: #4f8ef7; }
      `}</style>
    </div>
  )
}

function Badge({ bg, color, children }: { bg: string; color: string; children: React.ReactNode }) {
  return (
    <span style={{ background: bg, color }} className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold">
      {children}
    </span>
  )
}
