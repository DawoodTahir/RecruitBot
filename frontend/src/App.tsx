import ChatLayout from "./components/ChatLayout";
import SidePanel from "./components/SidePanel";
import { useChat } from "./hooks/useChat";

const App = () => {
  const chat = useChat();
  const {
    state,
    sendMessage,
    selectSession,
    createSession,
    setTemperature,
    setComposerValue,
    composerValue,
    temperature,
    selectedTools,
    toggleTool,
    uploadDocument,
    sendMutation,
    uploadMutation
  } = chat;

  return (
    <div className="min-h-screen w-full bg-slate-950 text-white py-6 px-4 md:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 lg:flex-row">
        <div className="flex-1 rounded-3xl border border-white/5 bg-white/5 p-1 shadow-elevated backdrop-blur-xl">
          <ChatLayout
            messages={state.messages}
            isSending={sendMutation.isPending}
            composerValue={composerValue}
            setComposerValue={setComposerValue}
            onSend={sendMessage}
            temperature={temperature}
            onTemperatureChange={setTemperature}
            selectedTools={selectedTools}
          />
        </div>
        <aside className="w-full lg:w-96">
          <SidePanel
            sessions={state.sessions}
            activeSessionId={state.activeSessionId}
            onSelectSession={selectSession}
            onCreateSession={createSession}
            tools={state.tools}
            selectedTools={selectedTools}
            toggleTool={toggleTool}
            onUpload={uploadDocument}
            uploadState={{ isPending: uploadMutation.isPending }}
            isLoading={state.isLoading}
          />
        </aside>
      </div>
    </div>
  );
};

export default App;

