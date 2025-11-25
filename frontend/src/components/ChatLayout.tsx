import { motion } from "framer-motion";
import { Activity, Gauge, Sparkles } from "lucide-react";
import MessageList from "./MessageList";
import Composer from "./Composer";
import type { ChatMessage } from "../types";

interface ChatLayoutProps {
  messages: ChatMessage[];
  isSending: boolean;
  composerValue: string;
  setComposerValue: (value: string) => void;
  onSend: (message: string) => Promise<void> | void;
  temperature: number;
  onTemperatureChange: (value: number) => void;
  selectedTools: string[];
}

const stats = [
  { label: "Context tokens", value: "32K", icon: Gauge },
  { label: "Latency (p95)", value: "2.1s", icon: Activity }
];

const ChatLayout = ({
  messages,
  composerValue,
  setComposerValue,
  onSend,
  isSending,
  temperature,
  onTemperatureChange,
  selectedTools
}: ChatLayoutProps) => {
  const handleSend = async () => {
    if (!composerValue.trim() || isSending) return;
    await onSend(composerValue.trim());
  };

  return (
    <div className="flex h-full flex-col rounded-[26px] border border-white/5 bg-slate-950/70 p-6">
      <div className="mb-6 flex flex-wrap items-center gap-4">
        <div className="inline-flex items-center gap-2 rounded-full border border-brand-400/30 bg-brand-500/10 px-4 py-1 text-sm font-medium text-brand-100">
          <Sparkles className="size-4 text-brand-200" />
          MCP Agent Live
        </div>
        <div className="flex flex-wrap gap-3 text-xs text-white/70">
          {selectedTools.length ? (
            <span className="rounded-full border border-white/10 px-3 py-1">
              Tools: {selectedTools.length}
            </span>
          ) : (
            <span className="rounded-full border border-rose-500/40 px-3 py-1 text-rose-200/80">
              No tools armed
            </span>
          )}
        </div>
        <div className="ml-auto flex gap-3">
          {stats.map((item) => (
            <motion.div
              key={item.label}
              className="rounded-2xl border border-white/5 bg-white/5 px-4 py-3"
              whileHover={{ translateY: -2 }}
            >
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-white/60">
                <item.icon className="size-4 text-brand-200" />
                {item.label}
              </div>
              <p className="text-xl font-semibold text-white">{item.value}</p>
            </motion.div>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-hidden rounded-2xl border border-white/5 bg-slate-950/40">
        <MessageList messages={messages} />
      </div>

      <Composer
        value={composerValue}
        onChange={setComposerValue}
        onSubmit={handleSend}
        disabled={isSending}
        temperature={temperature}
        onTemperatureChange={onTemperatureChange}
      />
    </div>
  );
};

export default ChatLayout;

