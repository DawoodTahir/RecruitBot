import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createSession,
  fetchMessages,
  fetchSessions,
  fetchTools,
  sendMessage,
  toggleTool,
  uploadDocument
} from "../lib/api";
import type { AgentTool, ChatMessage, ChatSession } from "../types";

interface SendArgs {
  message: string;
}

export const useChat = () => {
  const queryClient = useQueryClient();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.3);
  const [selectedTools, setSelectedTools] = useState<string[]>([]);
  const [composerValue, setComposerValue] = useState("");

  const sessionsQuery = useQuery<ChatSession[]>({
    queryKey: ["sessions"],
    queryFn: fetchSessions
  });

  useEffect(() => {
    if (!activeSessionId && sessionsQuery.data?.length) {
      setActiveSessionId(sessionsQuery.data[0].id);
    }
  }, [activeSessionId, sessionsQuery.data]);

  const toolsQuery = useQuery<AgentTool[]>({
    queryKey: ["tools"],
    queryFn: fetchTools,
    onSuccess: (data) => {
      setSelectedTools(data.filter((tool) => tool.enabled).map((tool) => tool.id));
    }
  });

  const messagesQuery = useQuery<ChatMessage[]>({
    queryKey: ["messages", activeSessionId],
    queryFn: () => fetchMessages(activeSessionId as string),
    enabled: Boolean(activeSessionId),
    refetchInterval: 10_000
  });

  const sendMutation = useMutation({
    mutationFn: (payload: SendArgs) => {
      if (!activeSessionId) {
        throw new Error("No active session.");
      }
      return sendMessage(activeSessionId, {
        message: payload.message,
        temperature,
        tools: selectedTools
      });
    },
    onMutate: async (payload) => {
      if (!activeSessionId) return;
      await queryClient.cancelQueries({ queryKey: ["messages", activeSessionId] });
      const previous = queryClient.getQueryData<ChatMessage[]>(["messages", activeSessionId]) ?? [];
      const optimistic: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: payload.message,
        createdAt: new Date().toISOString(),
        pending: true
      };
      queryClient.setQueryData(["messages", activeSessionId], [...previous, optimistic]);
      setComposerValue("");
      return { previous };
    },
    onError: (_err, _payload, context) => {
      if (context?.previous && activeSessionId) {
        queryClient.setQueryData(["messages", activeSessionId], context.previous);
      }
    },
    onSettled: () => {
      if (activeSessionId) {
        queryClient.invalidateQueries({ queryKey: ["messages", activeSessionId] });
      }
    }
  });

  const newSessionMutation = useMutation({
    mutationFn: createSession,
    onSuccess: (session) => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setActiveSessionId(session.id);
    }
  });

  const toggleToolMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => toggleTool(id, enabled),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["tools"] })
  });

  const uploadMutation = useMutation({
    mutationFn: uploadDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tools"] });
    }
  });

  const activeMessages = messagesQuery.data ?? [];

  const state = useMemo(
    () => ({
      sessions: sessionsQuery.data ?? [],
      messages: activeMessages,
      tools: toolsQuery.data ?? [],
      activeSessionId,
      temperature,
      composerValue,
      selectedTools,
      isLoading:
        sessionsQuery.isLoading ||
        messagesQuery.isLoading ||
        toolsQuery.isLoading ||
        sendMutation.isPending
    }),
    [
      sessionsQuery.data,
      activeMessages,
      toolsQuery.data,
      activeSessionId,
      temperature,
      composerValue,
      selectedTools,
      sessionsQuery.isLoading,
      messagesQuery.isLoading,
      toolsQuery.isLoading,
      sendMutation.isPending
    ]
  );

  return {
    state,
    sendMessage: (message: string) => sendMutation.mutateAsync({ message }),
    createSession: () => newSessionMutation.mutate(),
    selectSession: (sessionId: string) => setActiveSessionId(sessionId),
    setTemperature,
    setComposerValue,
    composerValue,
    temperature,
    selectedTools,
    toggleTool: (id: string, enabled: boolean) => {
      setSelectedTools((prev) =>
        enabled ? Array.from(new Set([...prev, id])) : prev.filter((tool) => tool !== id)
      );
      toggleToolMutation.mutate({ id, enabled });
    },
    uploadDocument: (file: File) => uploadMutation.mutateAsync(file),
    toolsQuery,
    sendMutation,
    uploadMutation
  };
};

