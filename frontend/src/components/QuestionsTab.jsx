import { useState } from "react"

export default function QuestionsTab({ questions }) {
  const [copied, setCopied] = useState(null)

  const handleCopy = (text, index) => {
    navigator.clipboard.writeText(text)
    setCopied(index)
    setTimeout(() => setCopied(null), 2000)
  }

  if (!questions || questions.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-6 shadow-sm text-center text-gray-400">
        No questions generated.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-gray-500 mb-1">
        Bring these questions to your next appointment.
      </p>
      {questions.map((q, i) => (
        <div key={i} className="bg-white rounded-2xl px-5 py-4 shadow-sm flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <span className="text-emerald-500 font-bold text-sm flex-shrink-0 mt-0.5">
              {i + 1}.
            </span>
            <p className="text-gray-700 text-sm leading-relaxed">{q}</p>
          </div>
          <button
            onClick={() => handleCopy(q, i)}
            className="flex-shrink-0 text-xs px-3 py-1.5 border border-gray-200 rounded-lg text-gray-500 hover:bg-gray-50 transition-colors"
          >
            {copied === i ? "Copied!" : "Copy"}
          </button>
        </div>
      ))}
    </div>
  )
}