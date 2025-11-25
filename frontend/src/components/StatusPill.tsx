import { ComponentType } from "react";

interface StatusPillProps {
  label: string;
  value: string;
  icon: ComponentType<{ className?: string }>;
}

const StatusPill = ({ label, value, icon: Icon }: StatusPillProps) => (
  <div className="flex items-center gap-2 rounded-2xl border border-white/5 bg-white/5 px-4 py-2">
    <Icon className="size-4 text-brand-200" />
    <div className="text-xs uppercase tracking-wide text-white/50">{label}</div>
    <span className="text-sm font-semibold text-white">{value}</span>
  </div>
);

export default StatusPill;

