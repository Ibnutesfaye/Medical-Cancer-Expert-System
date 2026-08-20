interface Props { 
  onClose: () => void;
  isAdmin?: boolean;
}

export default function HelpModal({ onClose, isAdmin }: Props) {
  const allItems = [
    { icon: '💬', title: 'Ask Questions', desc: 'Type any cancer-related question in the chat input and press Enter.' },
    { icon: '🖼️', title: 'Image Analysis', desc: 'Click the attachment icon to upload a medical image for cancer detection.' },
    { icon: '🎤', title: 'Voice Input', desc: 'Click the microphone icon to speak your question instead of typing.' },
    { icon: '🔊', title: 'Text to Speech', desc: 'Click the Listen button on any AI response to hear it read aloud.' },
    { icon: '📄', title: 'Upload PDFs', desc: 'Admins can upload medical PDFs via the Admin Dashboard to expand the knowledge base.', adminOnly: true },
    { icon: '🧠', title: 'DeepThink', desc: 'Use the DeepThink button for more detailed, analytical responses.' },
    { icon: '🔍', title: 'Search', desc: 'Use the Search button to search Wikipedia and PubMed for information.' },
    { icon: '🗑️', title: 'Delete Chats', desc: 'Hover over a chat in the sidebar and click the trash icon to delete it.' },
  ]

  const items = allItems.filter(item => !item.adminOnly || isAdmin)

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-[#161616] border border-[#2a2a2a] rounded-2xl p-6 max-w-md w-full shadow-2xl max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()} style={{ fontFamily: 'Inter, sans-serif' }}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-white">Help</h2>
          <button onClick={onClose} className="text-[#666] hover:text-[#aaa] transition-colors">
            <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>

        <div className="space-y-3">
          {items.map(({ icon, title, desc }) => (
            <div key={title} className="flex gap-3 p-3 rounded-xl bg-[#1a1a1a] border border-[#2a2a2a]">
              <span className="text-xl flex-shrink-0">{icon}</span>
              <div>
                <p className="text-sm font-medium text-white">{title}</p>
                <p className="text-xs text-[#888] mt-0.5">{desc}</p>
              </div>
            </div>
          ))}
        </div>

        {isAdmin && (
          <div className="mt-4 p-3 rounded-xl bg-[#1a3a5f] border border-[#2a5a8f]">
            <p className="text-xs text-[#60a5fa]">
              Default login: <span className="font-mono">admin</span> / <span className="font-mono">admin123</span>
            </p>
          </div>
        )}

        <button onClick={onClose}
          className="w-full mt-4 py-2.5 rounded-xl bg-[#242424] text-[#ccc] hover:bg-[#2e2e2e] transition-colors text-sm">
          Close
        </button>
      </div>
    </div>
  )
}
