import ReactMarkdown from 'react-markdown'

interface Citation {
  document_name: string
  page_number: number
  chunk_text: string
  relevance_score: number
}

interface MessageBubbleProps {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  isStreaming?: boolean
}

export default function MessageBubble({ role, content, citations, isStreaming }: MessageBubbleProps) {
  const isUser = role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-3xl ${isUser ? 'bg-blue-600' : 'bg-gray-700'} rounded-lg px-4 py-3 shadow-lg`}>
        <div className="text-white prose prose-invert max-w-none">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
        
        {isStreaming && (
          <div className="flex items-center mt-2 text-gray-400 text-sm">
            <div className="animate-pulse">●</div>
            <span className="ml-2">Generating...</span>
          </div>
        )}
        
        {citations && citations.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-600">
            <div className="text-xs text-gray-400 mb-2">Sources:</div>
            {citations.map((citation, idx) => (
              <div key={idx} className="text-xs text-gray-300 mb-1">
                [{idx + 1}] {citation.document_name} (Page {citation.page_number})
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
