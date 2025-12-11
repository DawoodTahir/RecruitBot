import { Bot } from "lucide-react";

const TypingIndicator = () => {
    return (
        <div className="flex items-end gap-3 animate-fade-in">
            {/* Bot Avatar */}
            <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-white/10 border border-white/20">
                <Bot className="size-4 text-white/70" />
            </div>

            {/* Typing Bubble */}
            <div className="bg-white/5 border border-white/10 rounded-2xl rounded-bl-md px-4 py-3">
                <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
            </div>
        </div>
    );
};

export default TypingIndicator;
