import { useState, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || window.location.origin

interface User {
  id: number
  username: string
  email: string | null
  full_name: string | null
  is_admin: boolean
  is_active: boolean
  created_at: string
  updated_at: string
}

interface Document {
  id: number
  filename: string
  original_name: string
  file_size_bytes: number
  total_pages: number | null
  total_chunks: number
  status: string
  created_at: string
  uploaded_by: number
}

interface ImageAnalysis {
  id: number
  original_filename: string
  cancer_detected: boolean
  cancer_type: string | null
  confidence: number | null
  created_at: string
  user_id: number
  training_accuracy?: number
  validation_accuracy?: number
  training_loss?: number
  model_used?: string
  evaluation_info?: any
}

interface AdminPageProps {
  token: string
  onBack: () => void
}

export default function AdminPage({ token, onBack }: AdminPageProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'documents' | 'images'>('overview')
  const [users, setUsers] = useState<User[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const [images, setImages] = useState<ImageAnalysis[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [uploadingPDF, setUploadingPDF] = useState(false)

  useEffect(() => {
    if (activeTab === 'users') fetchUsers()
    else if (activeTab === 'documents') fetchDocuments()
    else if (activeTab === 'images') fetchImages()
    else if (activeTab === 'overview') fetchAll()
  }, [activeTab])

  const fetchAll = async () => {
    setLoading(true)
    await Promise.all([fetchUsers(), fetchDocuments(), fetchImages()])
    setLoading(false)
  }

  const fetchUsers = async () => {
    if (activeTab !== 'overview') setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_URL}/admin/users`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to fetch users')
      const data = await res.json()
      setUsers(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      if (activeTab !== 'overview') setLoading(false)
    }
  }

  const fetchDocuments = async () => {
    if (activeTab !== 'overview') setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_URL}/documents/`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to fetch documents')
      const data = await res.json()
      setDocuments(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      if (activeTab !== 'overview') setLoading(false)
    }
  }

  const fetchImages = async () => {
    if (activeTab !== 'overview') setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_URL}/images/admin/all`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to fetch image history')
      const data = await res.json()
      setImages(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      if (activeTab !== 'overview') setLoading(false)
    }
  }

  const handleDeleteUser = async (id: number) => {
    if (!confirm('Delete this user?')) return
    try {
      const res = await fetch(`${API_URL}/admin/users/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to delete user')
      fetchUsers()
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleDeleteDocument = async (id: number) => {
    if (!confirm('Delete this document?')) return
    try {
      const res = await fetch(`${API_URL}/documents/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Failed to delete document')
      fetchDocuments()
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleDeleteImage = async (id: number) => {
    if (!confirm('Are you sure you want to delete this image?')) return
    try {
      const res = await fetch(`${API_URL}/images/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) {
        const errorData = await res.json().catch(() => null)
        throw new Error(errorData?.detail || 'Failed to delete image analysis')
      }
      setImages(prev => prev.filter(i => i.id !== id))
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleUploadPDF = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (!file.name.endsWith('.pdf')) {
      setError('Only PDF files allowed')
      return
    }

    setUploadingPDF(true)
    setError('')
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_URL}/documents/ingest`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      })
      if (!res.ok) throw new Error('Failed to upload PDF')
      alert('PDF uploaded!')
      fetchDocuments()
    } catch (err: any) {
      setError(err.message)
    } finally {
      setUploadingPDF(false)
      e.target.value = ''
    }
  }

  const formatDate = (d: string) => new Date(d).toLocaleString()
  const formatBytes = (b: number) => {
    if (b < 1024) return b + ' B'
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB'
    return (b / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div className="h-screen w-screen overflow-y-auto bg-gray-50 dark:bg-[#0f0f0f] text-gray-900 dark:text-white transition-colors duration-300" style={{ fontFamily: 'Inter, sans-serif' }}>
      <header className="bg-white dark:bg-[#161616] border-b border-gray-200 dark:border-[#2a2a2a] px-6 py-4 flex-shrink-0 transition-colors duration-300">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button onClick={onBack} className="text-[#666] hover:text-[#aaa] transition-colors" title="Back">
              <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
            <h1 className="text-xl font-bold">Admin Dashboard</h1>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold text-white shadow-sm">A</div>
            <span className="text-sm text-gray-400 dark:text-[#aaa]">Admin</span>
          </div>
        </div>
      </header>

      <div className="bg-white dark:bg-[#161616] border-b border-gray-100 dark:border-[#2a2a2a] transition-all duration-300">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-6">
            {(['overview', 'users', 'documents', 'images'] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className={['px-4 py-3 text-sm font-medium border-b-2 transition-colors capitalize',
                  activeTab === tab ? 'border-blue-600 dark:border-[#4f8ef7] text-gray-900 dark:text-white' : 'border-transparent text-gray-400 dark:text-[#666] hover:text-gray-600 dark:hover:text-[#aaa]'].join(' ')}>
                {tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {error && <div className="mb-4 px-4 py-3 rounded-xl bg-[#2a1a1a] border border-[#5a2a2a] text-red-400 text-sm">{error}</div>}

        {loading ? <div className="text-center py-12 text-[#666]">Loading...</div> : (
          <>
            {activeTab === 'overview' && (
              <div>
                <h2 className="text-lg font-semibold mb-6">System Overview</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                  <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-6 shadow-sm">
                    <div className="text-gray-400 dark:text-[#666] text-sm mb-1">Total Users</div>
                    <div className="text-3xl font-bold text-gray-900 dark:text-white">{users.length}</div>
                  </div>
                  <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-6 shadow-sm">
                    <div className="text-gray-400 dark:text-[#666] text-sm mb-1">Documents</div>
                    <div className="text-3xl font-bold text-gray-900 dark:text-white">{documents.length}</div>
                  </div>
                  <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-100 dark:border-[#2a2a2a] p-6 shadow-sm">
                    <div className="text-gray-400 dark:text-[#666] text-sm mb-1">Image Analyses</div>
                    <div className="text-3xl font-bold text-gray-900 dark:text-white">{images.length}</div>
                  </div>
                </div>
                {users.length === 1 && documents.length === 0 && images.length === 0 && (
                  <div className="bg-[#1a2a3a] border border-[#2a4a6a] rounded-xl p-4 text-sm text-[#60a5fa] mb-8">
                    ℹ️ You're using main.py (in-memory mode). For full admin features with data persistence, use main_v2.py with MySQL.
                  </div>
                )}


              </div>
            )}

            {activeTab === 'users' && (
              <div>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-semibold">Users ({users.length})</h2>
                </div>
                <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-200 dark:border-[#2a2a2a] shadow-sm overflow-x-auto hide-scrollbar">
                  <table className="w-full">
                    <thead className="bg-gray-50 dark:bg-[#1a1a1a] border-b border-gray-100 dark:border-[#2a2a2a]">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">ID</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Username</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Email</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Role</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Status</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Created</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {users.map(u => (
                        <tr key={u.id} className="border-b border-gray-50 dark:border-[#2a2a2a] hover:bg-gray-50 dark:hover:bg-[#1a1a1a] transition-colors">
                          <td className="px-4 py-3 text-sm text-[#aaa]">{u.id}</td>
                          <td className="px-4 py-3 text-sm font-medium">{u.username}</td>
                          <td className="px-4 py-3 text-sm text-[#aaa]">{u.email || '-'}</td>
                          <td className="px-4 py-3 text-sm">
                            {u.is_admin ? <span className="px-2 py-0.5 rounded-full bg-[#1a3a5f] text-[#60a5fa] text-xs">Admin</span>
                              : <span className="px-2 py-0.5 rounded-full bg-[#1a2a1a] text-[#4ade80] text-xs">User</span>}
                          </td>
                          <td className="px-4 py-3 text-sm">
                            {u.is_active ? <span className="px-2 py-0.5 rounded-full bg-[#1a3a2a] text-[#4ade80] text-xs">Active</span>
                              : <span className="px-2 py-0.5 rounded-full bg-[#2a1a1a] text-[#ef4444] text-xs">Inactive</span>}
                          </td>
                          <td className="px-4 py-3 text-sm text-[#666]">{formatDate(u.created_at)}</td>
                          <td className="px-4 py-3 text-sm">
                            <button onClick={() => handleDeleteUser(u.id)} className="text-red-400 hover:text-red-300">
                              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {users.length === 0 && <div className="text-center py-12 text-[#666]">No users</div>}
                </div>
              </div>
            )}

            {activeTab === 'documents' && (
              <div>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-semibold">Documents ({documents.length})</h2>
                  <label className="px-4 py-2 rounded-xl bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 cursor-pointer shadow-md transition-all">
                    {uploadingPDF ? 'Uploading...' : '📄 Upload PDF'}
                    <input type="file" accept=".pdf" onChange={handleUploadPDF} disabled={uploadingPDF} className="hidden" />
                  </label>
                </div>
                <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-200 dark:border-[#2a2a2a] shadow-sm overflow-x-auto hide-scrollbar">
                  <table className="w-full">
                    <thead className="bg-gray-50 dark:bg-[#1a1a1a] border-b border-gray-100 dark:border-[#2a2a2a]">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">ID</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Filename</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Size</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Chunks</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Status</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Uploaded</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documents.map(d => (
                        <tr key={d.id} className="border-b border-gray-50 dark:border-[#2a2a2a] hover:bg-gray-50 dark:hover:bg-[#1a1a1a] transition-colors">
                          <td className="px-4 py-3 text-sm text-[#aaa]">{d.id}</td>
                          <td className="px-4 py-3 text-sm font-medium max-w-xs truncate" title={d.original_name}>{d.original_name}</td>
                          <td className="px-4 py-3 text-sm text-[#aaa]">{formatBytes(d.file_size_bytes)}</td>
                          <td className="px-4 py-3 text-sm text-[#aaa]">{d.total_chunks}</td>
                          <td className="px-4 py-3 text-sm">
                            {d.status === 'completed' ? <span className="px-2 py-0.5 rounded-full bg-[#1a3a2a] text-[#4ade80] text-xs">Ready</span>
                              : d.status === 'processing' ? <span className="px-2 py-0.5 rounded-full bg-[#2a2a1a] text-[#fbbf24] text-xs">Processing</span>
                                : <span className="px-2 py-0.5 rounded-full bg-[#2a1a1a] text-[#ef4444] text-xs">Failed</span>}
                          </td>
                          <td className="px-4 py-3 text-sm text-[#666]">{formatDate(d.created_at)}</td>
                          <td className="px-4 py-3 text-sm">
                            <button onClick={() => handleDeleteDocument(d.id)} className="text-red-400 hover:text-red-300">
                              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {documents.length === 0 && <div className="text-center py-12 text-[#666]">No documents</div>}
                </div>
              </div>
            )}

            {activeTab === 'images' && (
              <div>
                <h2 className="text-lg font-semibold mb-6">Image Analyses ({images.length})</h2>
                <div className="bg-white dark:bg-[#161616] rounded-xl border border-gray-200 dark:border-[#2a2a2a] shadow-sm overflow-x-auto hide-scrollbar">
                  <table className="w-full">
                    <thead className="bg-gray-50 dark:bg-[#1a1a1a] border-b border-gray-100 dark:border-[#2a2a2a]">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">ID</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Filename</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Cancer</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Type</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Confidence</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Model Used</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Analyzed</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-[#888] uppercase">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {images.map(i => (
                        <tr key={i.id} className="border-b border-gray-50 dark:border-[#2a2a2a] hover:bg-gray-50 dark:hover:bg-[#1a1a1a] transition-colors">
                          <td className="px-4 py-3 text-sm text-[#aaa]">{i.id}</td>
                          <td className="px-4 py-3 text-sm font-medium max-w-xs truncate" title={i.original_filename}>{i.original_filename}</td>
                          <td className="px-4 py-3 text-sm">
                            {i.cancer_detected ? <span className="px-2 py-0.5 rounded-full bg-[#2a1a1a] text-[#ef4444] text-xs">Yes</span>
                              : <span className="px-2 py-0.5 rounded-full bg-[#1a3a2a] text-[#4ade80] text-xs">No</span>}
                          </td>
                          <td className="px-4 py-3 text-sm text-[#aaa]">{i.cancer_type || '-'}</td>
                          <td className="px-4 py-3 text-sm text-[#aaa]">{i.confidence ? `${(i.confidence * 100).toFixed(1)}%` : '-'}</td>
                          <td className="px-4 py-3 text-sm text-[#aaa]">{i.model_used || 'ResNet18'}</td>
                          <td className="px-4 py-3 text-sm text-[#666]">{formatDate(i.created_at)}</td>
                          <td className="px-4 py-3 text-sm">
                            <button onClick={() => handleDeleteImage(i.id)} className="text-red-400 hover:text-red-300">
                              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {images.length === 0 && <div className="text-center py-12 text-[#666]">No analyses</div>}
                </div>
              </div>
            )}
          </>
        )}
      </div>
      <style>{`
        .hide-scrollbar::-webkit-scrollbar { display: none; }
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
        ::-webkit-scrollbar { display: none; }
        * { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  )
}
