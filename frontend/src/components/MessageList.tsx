import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

interface MessageListProps {
  messages: ChatMessage[];
  isSending?: boolean;
}

const MessageList = ({ messages, isSending }: MessageListProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (node) {
      node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
    }
  }, [messages, isSending]);

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-6" ref={containerRef}>
      {messages.length === 0 && (
        <div className="mt-20 text-center text-sm text-white/50">
          Start a conversation...
        </div>
      )}
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
      {isSending && <TypingIndicator />}
    </div>
  );
};

export default MessageList;


