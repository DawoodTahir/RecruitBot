import { FormEvent, ReactNode, useMemo, useState, useRef, DragEvent } from "react";
import { Loader2, Mic, MicOff, Send, Paperclip, Shield, FileText, X, Check, Camera, Image as ImageIcon } from "lucide-react";
import CameraModal from "./CameraModal";
import clsx from "clsx";

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void | Promise<void>;
  disabled?: boolean;
  onUploadResume?: (file: File) => Promise<void> | void;
  isUploadingResume?: boolean;
  hasResume?: boolean;
  inputMode?: "text" | "voice";
}

const Composer = ({
  value,
  onChange,
  onSubmit,
  disabled,
  onUploadResume,
  isUploadingResume,
  hasResume,
  inputMode = "text"
}: ComposerProps) => {
  const [isListening, setIsListening] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showCamera, setShowCamera] = useState(false);
  const [capturedImage, setCapturedImage] = useState<File | null>(null);
  const hasText = useMemo(() => value.trim().length > 0, [value]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  type SpeechRecognitionResultEvent = {
    results: {
      0: {
        0: {
          transcript: string;
        };
      };
    };
  };

  type SpeechRecognitionInstance = {
    lang: string;
    interimResults: boolean;
    maxAlternatives: number;
    onresult: ((event: SpeechRecognitionResultEvent) => void) | null;
    onerror: ((event: any) => void) | null;
    onend: (() => void) | null;
    start: () => void;
  };

  type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

  const getSpeechRecognition = (): SpeechRecognitionConstructor | null => {
    if (typeof window === "undefined") return null;
    const win = window as Window &
      typeof globalThis & {
        webkitSpeechRecognition?: SpeechRecognitionConstructor;
        SpeechRecognition?: SpeechRecognitionConstructor;
      };
    return win.SpeechRecognition ?? win.webkitSpeechRecognition ?? null;
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await onSubmit();
  };

  const handleVoiceInput = () => {
    const Recognition = getSpeechRecognition();
    if (!Recognition) {
      if (typeof window !== "undefined") {
        window.alert("Voice capture is not supported in this browser yet.");
      }
      return;
    }

    try {
      const recognition = new Recognition();
      recognition.lang = "en-US";
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;
      setIsListening(true);

      recognition.onresult = (event: SpeechRecognitionResultEvent) => {
        const transcript = event.results[0][0].transcript;
        const nextValue = value ? `${value} ${transcript}` : transcript;
        onChange(nextValue.trimStart());
        setIsListening(false);
        // Auto-send after voice input
        if (nextValue.trim()) {
          setTimeout(() => onSubmit(), 100);
        }
      };

      recognition.onerror = (event: any) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
        if (typeof window !== "undefined") {
          window.alert(`Voice error: ${event.error}`);
        }
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.start();
    } catch {
      setIsListening(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (onUploadResume) {
      setUploadedFileName(file.name);
      setUploadProgress(0);

      // Simulate progress animation
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 100);

      await onUploadResume(file);

      clearInterval(progressInterval);
      setUploadProgress(100);

      // Keep showing the file name after upload
      setTimeout(() => {
        setUploadProgress(0);
      }, 2000);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = async (e: DragEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.pdf') || file.name.endsWith('.doc') || file.name.endsWith('.docx') || file.name.endsWith('.txt'))) {
      await handleFileUpload(file);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={clsx(
        "mt-3 rounded-2xl border bg-white/5 p-3 transition-all duration-300",
        isDragOver
          ? "border-[#40E0D0]/50 bg-[#40E0D0]/10 scale-[1.01]"
          : "border-white/10"
      )}
    >
      {isDragOver && (
        <div className="absolute inset-0 flex items-center justify-center rounded-3xl bg-brand-500/20 backdrop-blur-sm z-10 pointer-events-none">
          <div className="text-center">
            <FileText className="size-12 text-brand-200 mx-auto mb-2" />
            <p className="text-brand-100 font-semibold">Drop your resume here</p>
          </div>
        </div>
      )}

      <textarea
        className="w-full resize-none rounded-xl border-0 bg-transparent p-3 text-sm text-white placeholder:text-white/40 focus:outline-none"
        rows={2}
        placeholder={
          inputMode === "voice"
            ? "For this question, please answer using voice input."
            : "Type your message or upload a resume…"
        }
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!disabled && value.trim()) {
              void onSubmit();
            }
          }
        }}
        disabled={disabled || inputMode === "voice"}
      />

      {/* File preview section */}
      {uploadedFileName && hasResume && (
        <div className="mt-3 flex items-center gap-3 rounded-xl bg-emerald-500/15 border border-emerald-400/30 px-4 py-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-emerald-500/20">
            <FileText className="size-4 text-emerald-300" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-emerald-100 truncate">{uploadedFileName}</p>
            <p className="text-xs text-emerald-200/70">Resume uploaded successfully</p>
          </div>
          <Check className="size-5 text-emerald-400" />
        </div>
      )}

      {/* Upload progress bar */}
      {isUploadingResume && uploadProgress > 0 && uploadProgress < 100 && (
        <div className="mt-3 rounded-xl bg-white/5 p-3 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-white/70">Uploading {uploadedFileName}...</span>
            <span className="text-xs text-brand-200">{uploadProgress}%</span>
          </div>
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-brand-400 to-brand-500 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Captured Image Preview */}
      {capturedImage && (
        <div className="relative mb-3 inline-block">
          <div className="group relative overflow-hidden rounded-xl border border-white/20">
            <img
              src={URL.createObjectURL(capturedImage)}
              alt="Captured"
              className="h-20 w-auto object-cover"
            />
            <button
              type="button"
              onClick={() => setCapturedImage(null)}
              className="absolute right-1 top-1 rounded-full bg-black/50 p-1 text-white opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100 hover:bg-rose-500"
            >
              <X className="size-3" />
            </button>
          </div>
        </div>
      )}

      <div className="mt-3 flex flex-col gap-3 md:flex-row md:items-center">

        <div className="flex items-center gap-2">
          {/* Camera Button */}
          <button
            type="button"
            onClick={() => setShowCamera(true)}
            className="inline-flex h-12 w-12 items-center justify-center rounded-2xl px-0 text-sm font-semibold uppercase tracking-wide transition-all duration-200 border border-white/10 bg-white/5 hover:border-white/30 hover:bg-white/10"
          >
            <Camera className="size-5" />
          </button>
          <button
            type="button"
            onClick={handleVoiceInput}
            disabled={disabled || isListening}
            className={clsx(
              "inline-flex h-12 items-center justify-center gap-2 rounded-2xl px-5 text-sm font-semibold uppercase tracking-wide transition-all duration-200",
              "border border-white/10 bg-white/5 hover:border-white/30 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50",
              isListening && "animate-pulse border-rose-400/50 bg-rose-500/10"
            )}
          >
            {isListening ? <MicOff className="size-4 text-rose-200" /> : <Mic className="size-4" />}
            {isListening ? "Listening…" : "Voice"}
          </button>
          <button
            type="submit"
            disabled={disabled || inputMode === "voice" || !value.trim()}
            className={clsx(
              "inline-flex h-12 items-center justify-center gap-2 rounded-2xl px-6 text-sm font-semibold uppercase tracking-wide transition-all duration-200",
              "border border-white/10 bg-white/5 hover:border-white/30 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50",
              value.trim() && !disabled && "bg-brand-500 border-transparent hover:bg-brand-400"
            )}
          >
            <Send className="size-4" />
            Send
          </button>
          <div className="relative">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploadingResume}
              className={clsx(
                "inline-flex h-12 items-center justify-center gap-2 rounded-2xl px-5 text-sm font-semibold uppercase tracking-wide transition-all duration-200",
                "border border-white/10 bg-white/5 hover:border-white/30 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50",
                hasResume && "border-emerald-400/30 bg-emerald-500/10"
              )}
            >
              {isUploadingResume ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Uploading…
                </>
              ) : (
                <>
                  <Paperclip className="size-4" />
                  Resume
                </>
              )}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.doc,.docx,.txt"
              className="hidden"
              onChange={async (event) => {
                const file = event.target.files?.[0];
                if (file) {
                  await handleFileUpload(file);
                }
                event.target.value = "";
              }}
            />
          </div>
        </div>
      </div>

      {showCamera && (
        <CameraModal
          onCapture={(file) => setCapturedImage(file)}
          onClose={() => setShowCamera(false)}
        />
      )}
    </form>
  );
};

export default Composer;



