export default function JargonTab({ jargon }) {
  if (!jargon || jargon.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-6 shadow-sm text-center text-gray-400">
        No medical terms found to explain.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {jargon.map((item, i) => (
        <div key={i} className="bg-white rounded-2xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="font-semibold text-gray-900">{item.term}</span>
            <span className={`text-xs px-2 py-1 rounded-full font-medium
              ${item.confidence >= 0.85
                ? "bg-emerald-100 text-emerald-700"
                : "bg-amber-100 text-amber-700"
              }`}>
              {Math.round(item.confidence * 100)}% confident
            </span>
          </div>
          <p className="text-gray-600 text-sm leading-relaxed">{item.explanation}</p>
          {item.source_sentence && (
            <p className="mt-2 text-xs text-gray-400 italic border-l-2 border-gray-200 pl-2">
              From document: "{item.source_sentence}"
            </p>
          )}
        </div>
      ))}
    </div>
  )
}