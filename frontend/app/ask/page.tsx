"use client"

import { useState, useRef, useEffect } from "react"
import Link from "next/link"

export default function AskAI() {
  const [messages, setMessages] = useState<any[]>([])
  const [question, setQuestion] = useState("")
  const [loading, setLoading] = useState(false)

  const chatEndRef = useRef<HTMLDivElement>(null)

  const sampleQuestions = [
    "Summarize this document",
    "What are the key points?",
    "Explain the main topic",
    "List important information",
  ]

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const askAI = async (q?: string) => {
    const userQuestion = q || question
    if (!userQuestion || loading) return

    setMessages(prev => [...prev, { role: "user", content: userQuestion }])
    setQuestion("")
    setLoading(true)

    try {
      const sessionId = localStorage.getItem("docmind_session_id")
      if (!sessionId) {
        alert("No session found. Please upload documents first.")
        setLoading(false)
        return
      }

      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/ask-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question: userQuestion
        })
      })

      if (!res.body) throw new Error("No response body")

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let text = ""

      setMessages(prev => [...prev, { role: "ai", content: "" }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        text += decoder.decode(value)

        setMessages(prev => {
          const updated = [...prev]
          updated[updated.length - 1].content = text
          return updated
        })
      }
    } catch {
      setMessages(prev => [...prev, { role: "ai", content: "⚠️ Error contacting AI server" }])
    }
    setLoading(false)
  }

  const handleKeyPress = (e: any) => {
    if (e.key === "Enter") {
      e.preventDefault()
      askAI()
    }
  }

  return (
    <div className="flex-1 flex flex-col items-center py-8 px-4 bg-gray-50 h-[calc(100vh-80px)]">
      <div className="w-full max-w-4xl bg-white rounded-3xl shadow-sm border border-gray-200 flex flex-col h-full overflow-hidden">
        
        {/* CHAT MESSAGES AREA */}
        <div className="flex-1 overflow-y-auto p-6 md:p-10 flex flex-col gap-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center fade-in">
              <div className="w-16 h-16 bg-red-50 text-airbnb rounded-full flex items-center justify-center text-3xl mb-6">💬</div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Ask about your document</h2>
              <p className="text-gray-500 mb-8 max-w-sm">
                Select a sample question below or type your own question to get started.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-lg">
                {sampleQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => askAI(q)}
                    className="p-4 border border-gray-200 rounded-xl text-left hover:border-airbnb hover:bg-red-50 transition text-gray-700 font-medium"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[85%] md:max-w-[75%] rounded-2xl px-6 py-4 ${
                msg.role === "user" 
                  ? "bg-airbnb text-white rounded-tr-sm" 
                  : "bg-gray-100 text-gray-800 rounded-tl-sm"
              }`}>
                <div className="leading-relaxed whitespace-pre-wrap">{msg.content}</div>
                {msg.role === "ai" && msg.content && (
                  <div className="mt-3 flex justify-end">
                    <button 
                      onClick={() => navigator.clipboard.writeText(msg.content)}
                      className="text-xs flex items-center gap-1 text-gray-500 hover:text-gray-800 transition"
                      title="Copy to clipboard"
                    >
                      <span>📋</span> Copy
                    </button>
                  </div>
                )}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-300/30 text-xs text-gray-500">
                    <span className="font-semibold">Sources:</span> {msg.sources.join(", ")}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-2xl rounded-tl-sm px-6 py-4 flex items-center gap-2 text-gray-500">
                <span className="animate-bounce">●</span>
                <span className="animate-bounce delay-100">●</span>
                <span className="animate-bounce delay-200">●</span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* INPUT AREA */}
        <div className="p-4 border-t border-gray-100 bg-white">
          <div className="max-w-4xl mx-auto flex items-center gap-3 bg-gray-50 border border-gray-200 rounded-full p-2 pl-6 focus-within:ring-2 focus-within:ring-airbnb/20 focus-within:border-airbnb transition">
            <input
              type="text"
              placeholder="Ask anything about the document..."
              className="flex-1 bg-transparent border-none outline-none text-gray-800 placeholder-gray-400 py-2"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={loading}
            />
            <button 
              onClick={() => askAI()}
              disabled={!question.trim() || loading}
              className="bg-airbnb hover:bg-airbnb-dark text-white rounded-full p-3 px-6 font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
        </div>

      </div>
    </div>
  )
}
