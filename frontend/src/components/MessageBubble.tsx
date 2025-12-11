import { Bot, Loader2, User } from "lucide-react";
import type { ChatMessage } from "../types";

interface MessageBubbleProps {
  message: ChatMessage;
}

const MessageBubble = ({ message }: MessageBubbleProps) => {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  return (
    <div
      className={`flex items-end gap-3 animate-fade-in ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${isUser
            ? "bg-[#40E0D0]/20 border border-[#40E0D0]/40"
            : isSystem
              ? "bg-amber-500/20 border border-amber-400/40"
              : "bg-white/10 border border-white/20"
          }`}
      >
        {isUser ? (
          <User className="size-4 text-[#40E0D0]" />
        ) : (
          <Bot className="size-4 text-white/70" />
        )}
      </div>

      {/* Message Bubble */}
      <div className={`relative max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
        {/* Bubble */}
        <div
          className={`rounded-2xl px-4 py-3 ${isUser
              ? "bg-[#40E0D0]/20 border border-[#40E0D0]/30 rounded-br-md"
              : isSystem
                ? "bg-amber-500/10 border border-amber-400/30 rounded-bl-md"
                : "bg-white/5 border border-white/10 rounded-bl-md"
            }`}
        >
          <p className="text-sm leading-relaxed text-white/90 whitespace-pre-wrap">
            {message.content}
          </p>
        </div>

        {/* Timestamp */}
        <div className={`flex items-center gap-2 mt-1 text-[10px] text-white/40 ${isUser ? "justify-end" : "justify-start"}`}>
          <span>{new Date(message.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          {message.pending && <Loader2 className="size-3 animate-spin" />}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;



