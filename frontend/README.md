# Chatbot Frontend

React + Vite control room for the MCP / WhatsApp chatbot. Highlights:

- Conversational workspace with streaming-friendly layout
- Session switcher + mission control sidebar
- Tool toggles mirrored from FastMCP
- Graph RAG upload shortcut

## Development

```bash
cd frontend
npm install
npm run dev
```

Expose the Python backend via `http://localhost:8000` or override with `VITE_API_BASE_URL`.

