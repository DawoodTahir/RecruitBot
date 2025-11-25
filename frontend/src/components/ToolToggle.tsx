import clsx from "clsx";
import type { AgentTool } from "../types";

interface ToolToggleProps {
  tool: AgentTool;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

const ToolToggle = ({ tool, checked, onChange }: ToolToggleProps) => (
  <label className="flex cursor-pointer flex-col gap-2 rounded-2xl border border-white/5 bg-white/5 p-4 transition hover:border-brand-400/40">
    <div className="flex items-center gap-2">
      <button
        type="button"
        className={clsx(
          "relative inline-flex h-6 w-11 items-center rounded-full border border-white/10",
          checked ? "bg-brand-500" : "bg-slate-700"
        )}
        onClick={() => onChange(!checked)}
      >
        <span
          className={clsx(
            "inline-block h-4 w-4 rounded-full bg-white transition",
            checked ? "translate-x-5" : "translate-x-1"
          )}
        />
      </button>
      <div className="flex-1">
        <p className="text-sm font-semibold text-white">{tool.label}</p>
        <p className="text-xs text-white/60">{tool.description}</p>
      </div>
    </div>
  </label>
);

export default ToolToggle;

