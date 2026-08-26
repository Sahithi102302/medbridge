export default function MedicationList({ medications }) {
  if (!medications || medications.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-6 shadow-sm text-center text-gray-400">
        No medications listed in this document.
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {medications.map((med, i) => (
        <div key={i} className="bg-white rounded-2xl p-5 shadow-sm">
          <div className="flex items-start justify-between mb-2">
            <div>
              <span className="font-semibold text-gray-900 capitalize">{med.name}</span>
              {med.brand_name && (
                <span className="text-gray-400 text-sm ml-2">({med.brand_name})</span>
              )}
            </div>
            <span className="text-xs px-2 py-1 bg-blue-50 text-blue-600 rounded-full">
              {med.frequency}
            </span>
          </div>
          <p className="text-gray-600 text-sm leading-relaxed">{med.purpose}</p>
          {med.warning && (
            <div className="mt-3 flex items-start gap-2 bg-red-50 rounded-lg px-3 py-2">
              <svg className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <p className="text-xs text-red-700">{med.warning}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}