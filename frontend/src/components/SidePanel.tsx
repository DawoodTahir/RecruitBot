import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { API_BASE_URL } from "../config";
import type { InterviewState } from "../types";
import { Camera, Lightbulb, Square, Shield, ChevronDown } from "lucide-react";

interface SidePanelProps {
  hasResume: boolean;
  onUpload: (file: File) => Promise<void> | void; // kept for props compatibility, not used
  isUploading: boolean; // kept for props compatibility, not used
  userId: string;
  interviewState: InterviewState | null;
}

interface CompanySummary {
  name: string;
  website: string | null;
  linkedin: string | null;
  services_summary: string;
  culture_summary: string;
}

interface SuggestionsPayload {
  role: string | null;
  hot_skills: string[];
  attitude_tips: string[];
  company?: CompanySummary | null;
}

const SuggestionBox = ({
  userId,
  interviewState
}: {
  userId: string;
  interviewState: InterviewState | null;
}) => {
  const { data, isLoading, error } = useQuery<SuggestionsPayload>({
    queryKey: ["suggestions", userId],
    // We can show company + default role suggestions even before CV upload,
    // so only gate on userId.
    enabled: Boolean(userId),
    queryFn: async () => {
      const params = new URLSearchParams({ user_id: userId });
      const res = await fetch(`${API_BASE_URL}/suggestions?${params.toString()}`);
      if (!res.ok) {
        throw new Error("Failed to load suggestions");
      }
      return (await res.json()) as SuggestionsPayload;
    }
  });

  const displayRole = data?.role || "this role";

  type Phase = "company" | "skills" | "tips";
  const [phase, setPhase] = useState<Phase>("company");
  const [headline, setHeadline] = useState("");
  const [visibleCount, setVisibleCount] = useState(0);
  const [tipsVisibleCount, setTipsVisibleCount] = useState(0);

  // Phase control: company info -> skills -> interview tips -> loop back
  useEffect(() => {
    if (!data) return;
    const hasCompany =
      !!data.company && (!!data.company.services_summary || !!data.company.culture_summary);

    // Initial phase
    setPhase(hasCompany ? "company" : "skills");
    setHeadline("");
    setVisibleCount(0);
    setTipsVisibleCount(0);

    // Define phase durations
    const companyDuration = 12000; // 12 seconds for company
    const skillsDuration = 15000; // 15 seconds for skills
    const tipsDuration = 15000;   // 15 seconds for tips

    let currentPhase: Phase = hasCompany ? "company" : "skills";

    const cyclePhases = () => {
      if (hasCompany) {
        // Cycle: company -> skills -> tips -> company ...
        if (currentPhase === "company") {
          currentPhase = "skills";
          setPhase("skills");
          return skillsDuration;
        } else if (currentPhase === "skills") {
          currentPhase = "tips";
          setPhase("tips");
          return tipsDuration;
        } else {
          currentPhase = "company";
          setPhase("company");
          return companyDuration;
        }
      } else {
        // Cycle: skills -> tips -> skills ...
        if (currentPhase === "skills") {
          currentPhase = "tips";
          setPhase("tips");
          return tipsDuration;
        } else {
          currentPhase = "skills";
          setPhase("skills");
          return skillsDuration;
        }
      }
    };

    // Start cycling after initial phase
    let timeoutId: number;
    const scheduleNext = (delay: number) => {
      timeoutId = window.setTimeout(() => {
        const nextDelay = cyclePhases();
        scheduleNext(nextDelay);
      }, delay);
    };

    // Schedule first transition
    const initialDelay = hasCompany ? companyDuration : skillsDuration;
    scheduleNext(initialDelay);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [data]);

  // Animate skills (headline + chips) when in skills phase
  useEffect(() => {
    if (!data || phase !== "skills") return;
    const full = `Did you know most ${displayRole} have these skills?`;
    let charIndex = 0;
    let skillsTimer: number | undefined;

    setHeadline("");
    setVisibleCount(0);

    const typeTimer = window.setInterval(() => {
      charIndex += 1;
      setHeadline(full.slice(0, charIndex));

      if (charIndex >= full.length) {
        window.clearInterval(typeTimer);
        const maxSkills = Math.min(data.hot_skills.length, 10);
        skillsTimer = window.setInterval(() => {
          setVisibleCount((prev) => {
            const next = Math.min(maxSkills, prev + 1);
            if (next >= maxSkills && skillsTimer) {
              window.clearInterval(skillsTimer);
            }
            return next;
          });
        }, 350);
      }
    }, 35);

    return () => {
      window.clearInterval(typeTimer);
      if (skillsTimer) {
        window.clearInterval(skillsTimer);
      }
    };
  }, [data, displayRole, phase]);

  // Animate tips popping in one by one when in tips phase
  useEffect(() => {
    if (!data || phase !== "tips") return;
    setTipsVisibleCount(0);
    const maxTips = Math.min(data.attitude_tips.length, 10);
    const timer = window.setInterval(() => {
      setTipsVisibleCount((prev) => {
        const next = Math.min(maxTips, prev + 1);
        if (next >= maxTips) {
          window.clearInterval(timer);
        }
        return next;
      });
    }, 350);

    return () => window.clearInterval(timer);
  }, [data, phase]);

  if (isLoading || !data) {
    return (
      <div className="rounded-3xl border border-white/10 bg-white/10 p-5 flex flex-col items-center justify-center min-h-[200px]">
        {/* Infinite loading spinner */}
        <div className="relative">
          <div className="w-10 h-10 rounded-full border-2 border-white/20"></div>
          <div className="absolute top-0 left-0 w-10 h-10 rounded-full border-2 border-transparent border-t-[#24B3A8] animate-spin"></div>
        </div>
        <p className="mt-3 text-xs text-white/50">Loading suggestions...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-3xl border border-rose-500/40 bg-rose-500/10 p-4 text-xs text-rose-100">
        Couldn&apos;t load suggestions right now.
      </div>
    );
  }

  const skillsToShow = data.hot_skills.slice(0, Math.min(visibleCount, 10));
  const tipsToShow = data.attitude_tips.slice(0, Math.min(tipsVisibleCount, 10));

  return (
    <div className="space-y-4 rounded-3xl border border-white/10 bg-white/15 p-5 text-sm text-white/80">
      <div className="flex items-center gap-2 text-white">
        <Lightbulb className="size-4 text-brand-200" />
        <h3 className="text-xs font-semibold uppercase tracking-wide">
          {phase === "company"
            ? "RecruitLens"
            : phase === "tips"
              ? "Interview Tips"
              : `Top skills · ${displayRole}`}
        </h3>
      </div>

      {phase === "company" &&
        data.company &&
        (data.company.services_summary || data.company.culture_summary) && (
          <div className="space-y-2 text-xs text-white/80">
            <p className="font-semibold text-sm">
              {data.company.name}
              {data.company.website && (
                <span className="ml-2 text-[10px] text-brand-200 break-all">
                  {data.company.website}
                </span>
              )}
            </p>
            {data.company.services_summary && (
              <p>
                <span className="font-semibold text-white/90">What they do: </span>
                <span className="text-white/80">{data.company.services_summary}</span>
              </p>
            )}
            {data.company.culture_summary && (
              <p>
                <span className="font-semibold text-white/90">Culture: </span>
                <span className="text-white/80">{data.company.culture_summary}</span>
              </p>
            )}
          </div>
        )}

      {phase === "skills" && data.hot_skills.length > 0 && (
        <div>
          <p className="mb-1 text-xs text-white/70">{headline}</p>
          <div className="flex flex-wrap gap-2">
            {skillsToShow.map((s) => (
              <span
                key={s}
                className="rounded-full bg-brand-500/15 px-3 py-1 text-xs text-white border border-brand-500/40"
              >
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {phase === "tips" && tipsToShow.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {tipsToShow.map((tip, i) => (
            <span
              key={tip}
              className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs text-emerald-100 border border-emerald-400/40 animate-fade-in"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              {tip}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

const CameraPreview = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const toggleCamera = async () => {
    if (isActive) {
      stopStream();
      setIsActive(false);
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("Camera not supported in this browser.");
      return;
    }

    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240 },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        void videoRef.current.play();
      }
      setIsActive(true);
    } catch (err) {
      setError("Unable to access camera.");
      setIsActive(false);
    }
  };

  useEffect(() => {
    // Auto-start camera on mount
    const startCamera = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError("Camera not supported in this browser.");
        return;
      }

      try {
        setError(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 320, height: 240 },
          audio: false
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play();
        }
        setIsActive(true);
      } catch (err) {
        setError("Unable to access camera.");
        setIsActive(false);
      }
    };

    startCamera();

    return () => {
      stopStream();
    };
  }, []);

  return (
    <div className="pointer-events-auto flex flex-col w-full gap-2">
      <div className="relative h-44 w-full overflow-hidden rounded-2xl border border-white/15 bg-[#24B3A8]/60">
        {isActive ? (
          <video
            ref={videoRef}
            className="h-full w-full object-cover"
            muted
            playsInline
          />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1 text-[10px] text-white/60">
            <Camera className="size-5 text-white" />
            <span>Camera off</span>
          </div>
        )}
        {error && (
          <div className="absolute inset-x-0 bottom-0 bg-rose-900/80 px-2 py-1 text-[9px] text-rose-100">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

const SidePanel = ({ hasResume, userId, interviewState }: SidePanelProps) => {
  const [privacyExpanded, setPrivacyExpanded] = useState(false);

  return (
    <div className="relative flex h-full flex-col gap-5 rounded-[26px] border border-white/10 bg-slate-900/60 p-5">
      {/* Camera at the top for video interview feel */}
      <div className="flex flex-col items-center gap-3">
        <CameraPreview />
        <p className="text-[10px] text-white/40 text-center">Your camera preview</p>
      </div>

      {/* Divider */}
      <div className="border-t border-white/10" />

      {/* Suggestions section */}
      <div>
        <h2 className="mb-3 text-sm font-semibold text-white/70 uppercase tracking-wide">Suggestions</h2>
        <SuggestionBox userId={userId} interviewState={interviewState} />
      </div>

      {/* Privacy section at bottom */}
      <div className="mt-auto border-t border-white/10 pt-3 text-[10px] leading-snug text-white/40">
        <button
          onClick={() => setPrivacyExpanded(!privacyExpanded)}
          className="flex items-center gap-2 text-[11px] text-white/50 hover:text-white/70 transition-colors w-full text-left"
        >
          <Shield className="size-3" />
          <span className="font-medium">Privacy & Security</span>
          <ChevronDown
            className={`size-3 ml-auto transition-transform duration-200 ${privacyExpanded ? "rotate-180" : ""
              }`}
          />
        </button>

        <div
          className={`overflow-hidden transition-all duration-300 ${privacyExpanded ? "max-h-40 opacity-100 mt-2" : "max-h-0 opacity-0"
            }`}
        >
          <div className="pt-2 pl-1 space-y-2 text-[10px] text-white/40">
            <p>Your resume and chat data are used only to personalize this interview experience.</p>
            <p>Data is processed securely and not shared with third parties.</p>
            <p>You can request data deletion at any time.</p>
          </div>
        </div>

        {!privacyExpanded && (
          <p className="mt-1 text-[10px] leading-snug text-white/40">
            Your data is secure. Click to learn more.
          </p>
        )}
      </div>
    </div>
  );
};

export default SidePanel;


