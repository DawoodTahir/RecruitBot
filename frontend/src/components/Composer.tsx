import { FormEvent } from "react";
import { Loader2, Send, ThermometerSun } from "lucide-react";
import clsx from "clsx";

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void | Promise<void>;
  disabled?: boolean;
  temperature: number;
  onTemperatureChange: (value: number) => void;
}

const Composer = ({
  value,
  onChange,
  onSubmit,
  disabled,
  temperature,
  onTemperatureChange
}: ComposerProps) => {
  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await onSubmit();
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="mt-4 rounded-3xl border border-white/10 bg-white/5 p-4 shadow-lg shadow-black/30"
    >
      <textarea
        className="w-full resize-none rounded-2xl border border-white/5 bg-black/30 p-4 text-base text-white placeholder:text-white/40 focus:border-brand-400/60 focus:outline-none"
        rows={4}
        placeholder="Ask for insight, trigger WhatsApp distribution, or upload a doc…"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            void onSubmit();
          }
        }}
        disabled={disabled}
      />

      <div className="mt-3 flex flex-col gap-3 md:flex-row md:items-center">
        <div className="flex flex-1 items-center gap-3 rounded-2xl border border-white/5 bg-black/30 px-4 py-3">
          <ThermometerSun className="size-5 text-brand-200" />
          <div className="flex flex-1 flex-col text-sm">
            <span className="text-white/70">Creativity</span>
            <span className="text-xs text-white/40">
              Current temperature {temperature.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={temperature}
            onChange={(event) => onTemperatureChange(Number(event.target.value))}
            className="h-1 w-32 accent-brand-400"
          />
        </div>
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className={clsx(
            "inline-flex items-center justify-center gap-2 rounded-2xl px-6 py-3 text-sm font-semibold uppercase tracking-wide transition",
            "bg-brand-500 hover:bg-brand-400 disabled:cursor-not-allowed disabled:bg-white/10"
          )}
        >
          {disabled ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              Thinking
            </>
          ) : (
            <>
              <Send className="size-4" />
              Send
            </>
          )}
        </button>
      </div>
    </form>
  );
};

export default Composer;

