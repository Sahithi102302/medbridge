export default function ProcessingScreen({ steps }) {
  const STEP_CONFIG = [
    { doneKey: "parsing_done",     label: "PDF Extracted",        icon: "📄" },
    { doneKey: "classifying_done", label: "Document Classified",  icon: "🔍" },
    { doneKey: "ner_done",         label: "Medical Terms Found",  icon: "🧬" },
    { doneKey: "chunking_done",    label: "Sections Identified",  icon: "📑" },
    { doneKey: "llm_done",         label: "Translation Complete", icon: "✨" },
  ]

  const doneKeys = new Set(steps.map(s => s.step))
  const doneCount = STEP_CONFIG.filter(s => doneKeys.has(s.doneKey)).length
  const progress = Math.round((doneCount / STEP_CONFIG.length) * 100)

  const getDetail = (key) => steps.find(s => s.step === key)?.detail || ""

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Med<span className="text-emerald-600">Bridge</span>
        </h1>
        <p className="text-gray-400 mt-1">Analyzing your document...</p>
      </div>

      <div className="w-full max-w-md bg-white rounded-2xl shadow-sm p-6">
        {/* Progress bar */}
        <div className="mb-6">
          <div className="flex justify-between text-xs text-gray-400 mb-2">
            <span>Processing</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2">
            <div
              className="bg-emerald-500 h-2 rounded-full transition-all duration-700"
              style={{ width: `${Math.max(progress, 5)}%` }}
            />
          </div>
        </div>

        {/* Steps */}
        <div className="flex flex-col gap-2">
          {STEP_CONFIG.map((config, index) => {
            const isDone = doneKeys.has(config.doneKey)
            const isActive = !isDone && doneCount === index
            const detail = getDetail(config.doneKey)

            return (
              <div
                key={config.doneKey}
                className={`flex items-center gap-3 p-3 rounded-xl transition-all
                  ${isDone ? "bg-emerald-50" : isActive ? "bg-blue-50" : "bg-gray-50"}`}
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
                  ${isDone ? "bg-emerald-100" : isActive ? "bg-blue-100" : "bg-gray-100"}`}
                >
                  {isDone ? (
                    <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : isActive ? (
                    <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <span className="text-xs text-gray-400 font-medium">{index + 1}</span>
                  )}
                </div>

                <div className="flex-1">
                  <p className={`text-sm font-medium
                    ${isDone ? "text-emerald-700" : isActive ? "text-blue-700" : "text-gray-400"}`}
                  >
                    {config.label}
                  </p>
                  {isDone && detail && (
                    <p className="text-xs text-gray-400 mt-0.5">{detail}</p>
                  )}
                </div>
                <span>{config.icon}</span>
              </div>
            )
          })}
        </div>
      </div>

      <p className="text-xs text-gray-400 mt-4">Usually takes 15–30 seconds</p>
    </div>
  )
}