import { Navigate, Route, Routes } from 'react-router-dom';
import { WorkspacePage } from './pages/WorkspacePage';
import { EditorPage } from './pages/EditorPage';

export function App() {
  return (
    <Routes>
      <Route path="/" element={<WorkspacePage />} />
      <Route path="/projects/:projectId" element={<EditorPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
