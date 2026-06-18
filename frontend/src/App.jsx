import { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    setSources([]);
    
    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (err) {
      setAnswer("Error connecting to server. Make sure it's running on port 8000.");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 20, fontFamily: "Arial" }}>
      <h1 style={{ color: "#2c3e50", textAlign: "center" }}>🕊️ Gandhi Knowledge Assistant</h1>
      <p style={{ color: "#7f8c8d", textAlign: "center", marginBottom: 30 }}>
        AI-Powered Digital Library & Research Assistant for Gandhian Literature
      </p>
      
      <div style={{ display: "flex", gap: 10, marginBottom: 30 }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="e.g., What is Satyagraha?"
          style={{ 
            flex: 1, 
            padding: "12px 16px", 
            fontSize: 16, 
            borderRadius: 8, 
            border: "2px solid #ddd",
            outline: "none"
          }}
        />
        <button 
          onClick={ask} 
          disabled={loading}
          style={{ 
            padding: "12px 24px", 
            background: loading ? "#95a5a6" : "#27ae60", 
            color: "white", 
            border: "none", 
            borderRadius: 8, 
            cursor: loading ? "not-allowed" : "pointer",
            fontSize: 16,
            fontWeight: "bold"
          }}
        >
          {loading ? "⏳ Thinking..." : "🔍 Ask"}
        </button>
      </div>

      {answer && (
        <div style={{ 
          background: "#e8f5e9", 
          padding: 20, 
          borderRadius: 12, 
          marginBottom: 20,
          border: "2px solid #27ae60"
        }}>
          <strong style={{ fontSize: 18, color: "#2c3e50" }}>📝 Answer:</strong>
          <p style={{ lineHeight: 1.8, fontSize: 16, marginTop: 10 }}>{answer}</p>
        </div>
      )}

      {sources.length > 0 && (
        <div>
          <h3 style={{ color: "#2c3e50" }}>📚 Sources:</h3>
          {sources.map((s, i) => (
            <div key={i} style={{ 
              borderLeft: "4px solid #27ae60", 
              padding: "12px 16px", 
              margin: "10px 0",
              background: "#fafafa",
              borderRadius: "0 8px 8px 0",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)"
            }}>
              <p style={{ margin: 0, fontWeight: "bold", color: "#2c3e50" }}>
                📖 {s.book} — Page {s.page}
              </p>
              <p style={{ color: "#555", fontStyle: "italic", marginTop: 5 }}>{s.snippet}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;