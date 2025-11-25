import { ChangeEvent, useRef } from "react";
import { CloudUpload, Loader2 } from "lucide-react";

interface DocumentUploadProps {
  onUpload: (file: File) => Promise<void> | void;
  isUploading: boolean;
}

const DocumentUpload = ({ onUpload, isUploading }: DocumentUploadProps) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await onUpload(file);
      event.target.value = "";
    }
  };

  return (
    <div className="rounded-2xl border border-dashed border-brand-400/40 bg-brand-500/10 p-4">
      <div className="flex items-center gap-3">
        <div className="rounded-2xl border border-brand-500/60 bg-brand-500/20 p-3">
          {isUploading ? (
            <Loader2 className="size-6 animate-spin text-brand-100" />
          ) : (
            <CloudUpload className="size-6 text-brand-100" />
          )}
        </div>
        <div className="flex-1 text-sm text-white/80">
          <p className="font-semibold text-white">Graph RAG</p>
          <p>Drop a PDF/Markdown file to sync into the knowledge graph.</p>
        </div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={isUploading}
          className="rounded-xl border border-brand-500/40 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-brand-50 hover:bg-brand-500/20"
        >
          {isUploading ? "Syncing…" : "Upload"}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.md,.txt"
          className="hidden"
          onChange={handleFile}
        />
      </div>
    </div>
  );
};

export default DocumentUpload;

