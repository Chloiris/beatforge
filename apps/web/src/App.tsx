import { Navigate, Route, Routes } from 'react-router-dom';
import { WorkspacePage } from './pages/WorkspacePage';
import { EditorPage } from './pages/EditorPage';
import { ChartEnginePage } from './pages/ChartEnginePage';
import { ChartGenerationPage } from './pages/ChartGenerationPage';

export function App() {
  return (
    <Routes>
      <Route path="/" element={<WorkspacePage />} />
      <Route path="/chart-engine" element={<ChartEnginePage />} />
      <Route path="/projects/:projectId/chart" element={<ChartGenerationPage />} />
      <Route path="/projects/:projectId" element={<EditorPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
