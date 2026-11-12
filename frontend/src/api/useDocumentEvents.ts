import { useEffect, useRef } from "react";

export interface DocumentEvent {
  type: "document_status" | "connected" | "pong";
  document_id?: string;
  filename?: string;
  status?: string;
  progress?: number;
  message?: string | null;
}

/**
 * Подписка на WebSocket-канал статусов обработки.
 * URL относительный (/api/v1/ws/...), поэтому работает и за reverse-proxy,
 * и в preview-окружении без обращения к localhost из браузера.
 */
export function useDocumentEvents(onEvent: (event: DocumentEvent) => void, token: string | null) {
  const handler = useRef(onEvent);
  handler.current = onEvent;

  useEffect(() => {
    if (!token) return;
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let pingTimer: number | undefined;
    let closed = false;

    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const url = `${protocol}://${window.location.host}/api/v1/ws/documents?token=${encodeURIComponent(token)}`;
      socket = new WebSocket(url);

      socket.onopen = () => {
        pingTimer = window.setInterval(() => socket?.readyState === WebSocket.OPEN && socket.send("ping"), 25_000);
      };
      socket.onmessage = (message) => {
        try {
          const data = JSON.parse(message.data) as DocumentEvent;
          if (data.type === "document_status") handler.current(data);
        } catch {
          /* ignore malformed frames */
        }
      };
      socket.onclose = () => {
        window.clearInterval(pingTimer);
        if (!closed) reconnectTimer = window.setTimeout(connect, 4000);
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(reconnectTimer);
      window.clearInterval(pingTimer);
      socket?.close();
    };
  }, [token]);
}
