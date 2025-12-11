import logo from "./assets/logo.svg";
import ChatLayout from "./components/ChatLayout";
import SidePanel from "./components/SidePanel";
import { useChat } from "./hooks/useChat";

const App = () => {
  const {
    messages,
    sendMessage,
    setComposerValue,
    composerValue,
    isSending,
    interviewState,
    userId,
    hasResume,
    uploadResume,
    isUploadingResume,
    inputMode
  } = useChat();

  return (
    <div className="h-screen w-full overflow-hidden bg-slate-900 text-white">
      <header className="w-full border-b border-white/10 bg-gradient-to-r from-[#40E0D0] to-teal-400 text-sm text-white shadow-lg shadow-teal-900/20">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 md:px-8">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center h-12 w-12 rounded-full bg-white backdrop-blur-sm border border-white/20 shadow-lg shadow-teal-900/20 overflow-hidden p-1.5 transition-transform hover:scale-105">
              <img src={logo} alt="RecruitLens Logo" className="h-full w-full object-contain" />
            </div>
            <span className="text-lg font-bold tracking-wide text-white drop-shadow-sm">RecruitLens</span>
          </div>
          <nav className="flex items-center gap-6 text-xs font-medium uppercase tracking-wide">
            <button className="relative py-1 transition-all duration-200 hover:text-teal-100 after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-white/50 after:transition-all after:duration-200 hover:after:w-full">About</button>
            <button className="relative py-1 transition-all duration-200 hover:text-teal-100 after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-white/50 after:transition-all after:duration-200 hover:after:w-full">Pricing</button>
            <button className="relative py-1 transition-all duration-200 hover:text-teal-100 after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-0 after:bg-white/50 after:transition-all after:duration-200 hover:after:w-full">Support</button>
            <button className="rounded-full bg-white px-5 py-2 text-xs font-bold text-[#40E0D0] shadow-md shadow-teal-900/10 transition-all duration-200 hover:bg-teal-50 hover:shadow-lg hover:scale-105">
              Login / Signup
            </button>
          </nav>
        </div>
      </header>
      <div className="mx-auto flex h-[calc(100vh-56px)] max-w-7xl flex-col gap-6 p-4 md:p-8 lg:flex-row">
        <div className="flex-1 min-h-0 rounded-3xl border border-white/10 bg-white/10 p-1 shadow-elevated backdrop-blur-xl">
          <ChatLayout
            messages={messages}
            isSending={isSending}
            composerValue={composerValue}
            setComposerValue={setComposerValue}
            onSend={sendMessage}
            interviewState={interviewState}
            hasResume={hasResume}
            onUploadResume={uploadResume}
            isUploadingResume={isUploadingResume}
            inputMode={inputMode}
          />
        </div>
        <aside className="w-full max-h-full overflow-y-auto rounded-3xl border border-white/10 bg-white/10 p-1 lg:w-80 xl:w-96">
          <SidePanel
            hasResume={hasResume}
            onUpload={uploadResume}
            isUploading={isUploadingResume}
            userId={userId}
            interviewState={interviewState}
          />
        </aside>
      </div>
    </div>
  );
};

export default App;

