import { useState } from "react"
import UploadScreen from "./components/UploadScreen"
import ProcessingScreen from "./components/ProcessingScreen"
import ResultsDashboard from "./components/ResultsDashboard"

export default function App() {
  const [screen, setScreen] = useState("upload") // upload | processing | results
  const [result, setResult] = useState(null)
  const [steps, setSteps] = useState([])

  const handleFileUpload = async (file) => {
    setScreen("processing")
    setSteps([])

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch("http://localhost:8000/analyze/stream", {
        method: "POST",
        body: formData,
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split("\n")

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event = JSON.parse(line.slice(6))

              if (event.step && event.detail) {
                console.log("Event received:", event.step, event.detail)
                setSteps(prev => [...prev, { step: event.step, detail: event.detail }])
              }

              if (event.done && event.result) {
                setResult(event.result)
                setScreen("results")
              }

              if (event.done && event.step === "error") {
                alert("Analysis failed: " + event.detail)
                setScreen("upload")
              }
            } catch (e) {
              // skip malformed events
            }
          }
        }
      }
    } catch (err) {
      alert("Could not connect to the API. Make sure the backend is running.")
      setScreen("upload")
    }
  }

  const handleReset = () => {
    setScreen("upload")
    setResult(null)
    setSteps([])
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {screen === "upload" && (
        <UploadScreen onUpload={handleFileUpload} />
      )}
      {screen === "processing" && (
        <ProcessingScreen steps={steps} />
      )}
      {screen === "results" && (
        <ResultsDashboard result={result} onReset={handleReset} />
      )}
    </div>
  )
}