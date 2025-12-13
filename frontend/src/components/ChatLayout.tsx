import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Waves } from "lucide-react";
import MessageList from "./MessageList";
import Composer from "./Composer";
import type { ChatMessage, InterviewState } from "../types";

interface ChatLayoutProps {
  messages: ChatMessage[];
  isSending: boolean;
  composerValue: string;
  setComposerValue: (value: string) => void;
  onSend: (message: string) => Promise<void> | void;
  interviewState: InterviewState | null;
  hasResume: boolean;
  onUploadResume: (file: File) => Promise<void> | void;
  isUploadingResume: boolean;
  inputMode: "text" | "voice";
}

const ChatLayout = ({
  messages,
  composerValue,
  setComposerValue,
  onSend,
  isSending,
  interviewState,
  hasResume,
  onUploadResume,
  isUploadingResume,
  inputMode
}: ChatLayoutProps) => {
  const [introText, setIntroText] = useState("");

  const handleSend = async () => {
    if (!composerValue.trim() || isSending || !hasResume) return;
    await onSend(composerValue.trim());
  };

  // Animated intro text when no resume is uploaded yet
  const introFullText =
    "Upload your resume once to unlock the interview. Use the Resume button beside the send control—after that, typing and voice both work.";

  useEffect(() => {
    if (!hasResume) {
      setIntroText("");
      let index = 0;
      const interval = window.setInterval(() => {
        index += 1;
        setIntroText(introFullText.slice(0, index));
        if (index >= introFullText.length) {
          window.clearInterval(interval);
        }
      }, 25);
      return () => window.clearInterval(interval);
    } else {
      // Reset when resume is uploaded
      setIntroText("");
    }
    // we intentionally depend only on hasResume to restart animation when it changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasResume]);

  return (
    <div className="flex h-full flex-col rounded-[26px] border border-white/10 bg-slate-900/60 p-6">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="inline-flex items-center gap-2 rounded-full border border-brand-400/40 bg-brand-500/15 px-4 py-1 text-sm font-medium text-brand-50">
          RecruitLens
        </div>
        <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-4 py-1 text-xs uppercase tracking-wide text-white/70">
          {isSending ? "Responding…" : hasResume ? "Ready" : "Waiting for resume"}
        </div>
        {inputMode === "voice" && (
          <div className="inline-flex items-center gap-2 rounded-full border border-amber-300/40 bg-amber-400/15 px-4 py-1 text-xs font-semibold uppercase tracking-wide text-amber-100">
            <Waves className="size-4" />
            Voice answer required
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-hidden rounded-2xl border border-white/10 bg-slate-900/30">
        {!hasResume ? (
          <div className="flex h-full flex-col items-center justify-center px-8 text-center">
            <p className="max-w-md text-base text-white/70 leading-relaxed">
              <span className="block text-xl font-medium text-brand-100 mb-3">Welcome to RecruitLens</span>
              Upload your resume to unlock the interview. <br />
              Use the <span className="text-[#24B3A8] font-medium">Resume</span> button below to get started.
            </p>
          </div>
        ) : (
          <div className="relative h-full">
            <MessageList messages={messages} isSending={isSending} />
            {inputMode === "voice" && (
              <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-center bg-gradient-to-b from-slate-900/80 to-transparent px-4 py-3 text-center text-xs font-medium uppercase tracking-wide text-amber-100">
                Please answer this question using the Voice button below.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Quick Action Buttons */}
      {hasResume && messages.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          <QuickActionButton
            label="Interview Tips"
            onClick={() => {
              setComposerValue("Give me some interview tips for my role");
              onSend("Give me some interview tips for my role");
            }}
          />
          <QuickActionButton
            label="Highlight Skills"
            onClick={() => {
              setComposerValue("What are my strongest skills from my resume?");
              onSend("What are my strongest skills from my resume?");
            }}
          />
          <QuickActionButton
            label="Practice Question"
            onClick={() => {
              setComposerValue("Give me a practice interview question");
              onSend("Give me a practice interview question");
            }}
          />
        </div>
      )}

      <Composer
        value={composerValue}
        onChange={setComposerValue}
        onSubmit={handleSend}
        disabled={isSending || !hasResume}
        onUploadResume={onUploadResume}
        isUploadingResume={isUploadingResume}
        hasResume={hasResume}
        inputMode={inputMode}
      />
    </div>
  );
};

// Quick Action Button Component
const QuickActionButton = ({ label, onClick }: { label: string; onClick: () => void }) => (
  <button
    type="button"
    onClick={onClick}
    className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/70 transition-all duration-200 hover:bg-white/10 hover:text-white hover:border-white/25 hover:scale-105"
  >
    {label}
  </button>
);

export default ChatLayout;

