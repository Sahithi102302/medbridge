import { useState } from "react"
import UrgencyPanel from "./UrgencyPanel"
import MedicationList from "./MedicationList"
import JargonTab from "./JargonTab"
import QuestionsTab from "./QuestionsTab"

export default function ResultsDashboard({ result, onReset }) {
  const [activeTab, setActiveTab] = useState("summary")

  const tabs = [
    { id: "summary", label: "Summary" },
    { id: "medications", label: `Medications (${result.medications?.length || 0})` },
    { id: "jargon", label: `Jargon (${result.jargon?.length || 0})` },
    { id: "questions", label: `Questions (${result.questions?.length || 0})` },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top bar */}
      <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">
            Med<span className="text-emerald-600">Bridge</span>
          </h1>
          <p className="text-xs text-gray-400 capitalize">{result.doc_type} document</p>
        </div>
        <button
          onClick={onReset}
          className="text-sm px-4 py-2 border border-gray-200 rounded-lg text-gray-600 hover:bg-gray-50"
        >
          Analyze another
        </button>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* Confidence badge */}
        <div className="flex items-center gap-2 mb-4">
          <span className="text-xs px-2 py-1 bg-emerald-100 text-emerald-700 rounded-full font-medium">
            {Math.round(result.overall_confidence * 100)}% confidence
          </span>
          <span className="text-xs text-gray-400 capitalize">
            {result.doc_type} summary
          </span>
        </div>

        {/* Urgency flags — always visible */}
        {result.urgency_flags?.length > 0 && (
          <UrgencyPanel flags={result.urgency_flags} />
        )}

        {/* Tabs */}
        <div className="flex gap-1 border-b border-gray-200 mb-6 mt-6">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors
                ${activeTab === tab.id
                  ? "border-emerald-500 text-emerald-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === "summary" && (
          <div className="bg-white rounded-2xl p-6 shadow-sm">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Plain English Summary
            </h2>
            <p className="text-gray-800 leading-relaxed text-lg">
              {result.summary}
            </p>
          </div>
        )}

        {activeTab === "medications" && (
          <MedicationList medications={result.medications} />
        )}

        {activeTab === "jargon" && (
          <JargonTab jargon={result.jargon} />
        )}

        {activeTab === "questions" && (
          <QuestionsTab questions={result.questions} />
        )}
      </div>
    </div>
  )
}