import { useCallback } from "react"
import { useDropzone } from "react-dropzone"

export default function UploadScreen({ onUpload }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0])
    }
  }, [onUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
  })

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      {/* Header */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          Med<span className="text-emerald-600">Bridge</span>
        </h1>
        <p className="text-gray-500 text-lg">
          Your medical documents, in plain English
        </p>
      </div>

      {/* Upload zone */}
      <div
        {...getRootProps()}
        className={`w-full max-w-lg border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all
          ${isDragActive
            ? "border-emerald-500 bg-emerald-50"
            : "border-gray-300 bg-white hover:border-emerald-400 hover:bg-gray-50"
          }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          {isDragActive ? (
            <p className="text-emerald-600 font-medium text-lg">Drop your PDF here</p>
          ) : (
            <>
              <p className="text-gray-700 font-medium text-lg">
                Drop your medical PDF here
              </p>
              <p className="text-gray-400 text-sm">
                or click to browse
              </p>
            </>
          )}
        </div>
      </div>

      {/* Supported types */}
      <div className="flex gap-2 mt-6 flex-wrap justify-center">
        {["Discharge Summary", "Lab Report", "Radiology Report", "Insurance EOB"].map(type => (
          <span key={type} className="text-xs px-3 py-1 bg-white border border-gray-200 rounded-full text-gray-500">
            {type}
          </span>
        ))}
      </div>

      <p className="text-xs text-gray-400 mt-4">
        Your document is never stored — processed in memory only
      </p>
    </div>
  )
}