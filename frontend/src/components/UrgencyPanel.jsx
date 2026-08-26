const SEVERITY_STYLES = {
  critical: "bg-red-50 border-red-200 text-red-800",
  soon: "bg-amber-50 border-amber-200 text-amber-800",
  routine: "bg-green-50 border-green-200 text-green-800",
}

const SEVERITY_DOT = {
  critical: "bg-red-500",
  soon: "bg-amber-500",
  routine: "bg-green-500",
}

export default function UrgencyPanel({ flags }) {
  return (
    <div className="flex flex-col gap-2 mb-2">
      {flags.map((flag, i) => (
        <div
          key={i}
          className={`flex items-start gap-3 px-4 py-3 rounded-xl border ${SEVERITY_STYLES[flag.severity] || SEVERITY_STYLES.routine}`}
        >
          <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${SEVERITY_DOT[flag.severity] || SEVERITY_DOT.routine}`} />
          <div>
            <p className="font-medium text-sm">{flag.text}</p>
            {flag.timeframe && (
              <p className="text-xs mt-0.5 opacity-75">{flag.timeframe}</p>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}