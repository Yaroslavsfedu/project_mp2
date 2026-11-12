import { useEffect, useState } from "react";
import { api, fileUrl } from "@/api/client";

/**
 * Загрузка изображения страницы с JWT-авторизацией.
 * Обычный <img src> не умеет передавать Authorization, поэтому файл
 * запрашивается как blob и подставляется через object URL.
 */
export function useAuthedImage(path: string | null | undefined) {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fullUrl = fileUrl(path);
    if (!fullUrl) {
      setUrl(null);
      return;
    }
    let objectUrl: string | null = null;
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .get(fullUrl.replace(/^\/api\/v1/, ""), { responseType: "blob" })
      .then((response) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(response.data as Blob);
        setUrl(objectUrl);
      })
      .catch((err) => {
        if (!cancelled) setError(err?.message ?? "Не удалось загрузить страницу");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);

  return { url, error, loading };
}
