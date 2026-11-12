import { Navigate, Route, Routes } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { AppLayout } from "@/components/layout/AppLayout";
import { ProtectedRoute } from "@/components/layout/ProtectedRoute";
import { Toaster } from "@/components/ui/Toaster";
import { useDocumentEvents } from "@/api/useDocumentEvents";
import { useAuth } from "@/store/auth";
import { toast } from "@/store/toast";
import { LoginPage } from "@/pages/LoginPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { DocumentsPage } from "@/pages/DocumentsPage";
import { UploadPage } from "@/pages/UploadPage";
import { DocumentViewPage } from "@/pages/DocumentViewPage";
import { SearchPage } from "@/pages/SearchPage";
import { AdminPage } from "@/pages/AdminPage";

function GlobalEvents() {
  const queryClient = useQueryClient();
  const token = useAuth((state) => state.token);

  useDocumentEvents((event) => {
    if (!event.document_id) return;
    queryClient.invalidateQueries({ queryKey: ["documents"] });
    queryClient.invalidateQueries({ queryKey: ["document", event.document_id] });
    queryClient.invalidateQueries({ queryKey: ["stats"] });

    if (event.status === "completed") {
      toast.success("Документ обработан", event.filename);
    } else if (event.status === "failed") {
      toast.error("Ошибка обработки", event.message ?? event.filename);
    }
  }, token);

  return null;
}

export default function App() {
  return (
    <>
      <GlobalEvents />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<AppLayout />}>
            <Route index element={<DashboardPage />} />
            <Route path="documents" element={<DocumentsPage />} />
            <Route path="upload" element={<UploadPage />} />
            <Route path="documents/:id" element={<DocumentViewPage />} />
            <Route path="search" element={<SearchPage />} />
            <Route path="admin" element={<AdminPage />} />
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Toaster />
    </>
  );
}
