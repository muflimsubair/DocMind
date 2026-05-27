"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"

export default function Home() {
  const [status, setStatus] = useState("")
  const [hasSession, setHasSession] = useState(false)
  const router = useRouter()

  useEffect(() => {
    setHasSession(Boolean(localStorage.getItem("docmind_session_id")))
  }, [])

  const uploadFiles = async (e: any) => {
    const files = e.target.files
    if (!files.length) return

    setStatus("Creating session...")

    try {
      const sessionRes = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/session`, {
        method: "POST",
      })

      const sessionData = await sessionRes.json()
      const sessionId = sessionData.session_id
      localStorage.setItem("docmind_session_id", sessionId)

      setStatus("Uploading documents...")

      for (let file of files) {
        const formData = new FormData()
        formData.append("file", file)

        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload?session_id=${sessionId}`, {
          method: "POST",
          body: formData,
        })
      }

      setStatus("✅ Documents uploaded successfully!")
      setHasSession(true)
    } catch {
      setStatus("❌ Upload failed. Please try again.")
    }
  }

  return (
    <div className="flex-1 flex items-center justify-center py-12 px-6">
      <div className="max-w-6xl w-full grid md:grid-cols-2 gap-16 items-center">
        
        {/* LEFT SECTION - HERO */}
        <div>
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-gray-900 leading-tight">
            Chat with your <br />
            <span className="text-airbnb">documents.</span>
          </h1>
          <p className="mt-6 text-xl text-gray-600 max-w-md leading-relaxed">
            Upload PDFs and get instant AI-powered answers with precise context. Secure, fast, and remarkably smart.
          </p>
        </div>

        {/* RIGHT SECTION - UPLOAD CARD */}
        <div className="bg-white rounded-3xl p-8 md:p-10 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)] border border-gray-100">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900">Upload Documents</h2>
            <p className="text-gray-500 mt-2">Select your PDFs to begin</p>
          </div>

          <label className="block border-2 border-dashed border-gray-300 rounded-2xl p-10 text-center cursor-pointer hover:border-airbnb hover:bg-red-50 transition duration-200 group">
            <div className="text-5xl mb-4 group-hover:scale-110 transition-transform duration-200">📄</div>
            <p className="text-gray-700 font-medium text-lg">Drag & drop your PDFs here</p>
            <p className="text-sm text-gray-400 mt-2">or click to browse from your computer</p>
            <input
              type="file"
              multiple
              className="hidden"
              onChange={uploadFiles}
              accept=".pdf"
            />
          </label>

          {status && (
            <div className={`mt-6 p-4 rounded-xl text-center text-sm font-medium ${status.includes('❌') ? 'bg-red-50 text-red-600' : 'bg-gray-50 text-gray-700'}`}>
              {status}
            </div>
          )}

          <button
            onClick={() => router.push("/ask")}
            disabled={!hasSession}
            className={`w-full mt-8 py-4 rounded-xl font-bold text-lg transition duration-200 ${
              hasSession
                ? "bg-airbnb hover:bg-airbnb-dark text-white shadow-lg hover:shadow-xl hover:-translate-y-0.5"
                : "bg-gray-100 text-gray-400 cursor-not-allowed"
            }`}
          >
            Start Chatting
          </button>
        </div>

      </div>
    </div>
  )
}
