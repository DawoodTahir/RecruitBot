import { Activity, PlugZap, Plus, Satellite } from "lucide-react";
import DocumentUpload from "./DocumentUpload";
import ToolToggle from "./ToolToggle";
import StatusPill from "./StatusPill";
import type { AgentTool, ChatSession } from "../types";

interface SidePanelProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  tools: AgentTool[];
  selectedTools: string[];
  toggleTool: (id: string, enabled: boolean) => void;
  onUpload: (file: File) => Promise<void> | void;
  uploadState: { isPending: boolean };
  isLoading: boolean;
}

const SidePanel = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onCreateSession,
  tools,
  selectedTools,
  toggleTool,
  onUpload,
  uploadState,
  isLoading
}: SidePanelProps) => {
  return (
    <div className="flex flex-col gap-5">
      <div className="rounded-3xl border border-white/5 bg-gradient-to-br from-white/5 to-transparent p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm uppercase tracking-wide text-white/50">Mission Control</p>
            <h2 className="text-xl font-semibold text-white">Conversation Stack</h2>
          </div>
          <button
            type="button"
            onClick={onCreateSession}
            className="inline-flex items-center gap-2 rounded-2xl border border-brand-500/40 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-brand-50"
          >
            <Plus className="size-4" />
            New Run
          </button>
        </div>
        <p className="mt-3 text-sm text-white/70">
          Spin up isolated sessions, attach context, and push results via WhatsApp.
        </p>
      </div>

      <section className="rounded-3xl border border-white/5 bg-white/5 p-4">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">
            Active Sessions
          </h3>
          <span className="text-xs text-white/50">
            {isLoading ? "Syncing…" : `${sessions.length} open`}
          </span>
        </div>
        <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
          {sessions.map((session) => {
            const selected = session.id === activeSessionId;
            return (
              <button
                key={session.id}
                onClick={() => onSelectSession(session.id)}
                className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                  selected
                    ? "border-brand-500/50 bg-brand-500/20 text-white"
                    : "border-white/5 bg-black/30 text-white/70 hover:border-white/20"
                }`}
              >
                <p className="text-sm font-medium">{session.title || "Untitled session"}</p>
                <p className="text-xs text-white/50">
                  {session.lastMessageAt
                    ? new Date(session.lastMessageAt).toLocaleString()
                    : "moments ago"}
                </p>
              </button>
            );
          })}
          {sessions.length === 0 && (
            <div className="rounded-2xl border border-dashed border-white/10 px-4 py-10 text-center text-sm text-white/60">
              No sessions yet. Launch the first run to begin.
            </div>
          )}
        </div>
      </section>

      <DocumentUpload onUpload={onUpload} isUploading={uploadState.isPending} />

      <section className="rounded-3xl border border-white/5 bg-white/5 p-4">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-white/70">
            Tooling
          </h3>
          <span className="text-xs text-white/50">{selectedTools.length} armed</span>
        </div>
        <div className="space-y-3">
          {tools.map((tool) => (
            <ToolToggle
              key={tool.id}
              tool={tool}
              checked={selectedTools.includes(tool.id)}
              onChange={(checked) => toggleTool(tool.id, checked)}
            />
          ))}
          {tools.length === 0 && (
            <p className="text-sm text-white/60">
              No MCP tools reported. Check that `FastMCP` server is running.
            </p>
          )}
        </div>
      </section>

      <div className="flex flex-wrap gap-3">
        <StatusPill label="MCP link" value="Stable" icon={Satellite} />
        <StatusPill label="Tool calls" value="Realtime" icon={PlugZap} />
        <StatusPill label="Observability" value="Nominal" icon={Activity} />
      </div>
    </div>
  );
};

export default SidePanel;

