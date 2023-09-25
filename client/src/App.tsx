import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import NavBar from './components/NavBar';
import Home from './pages/Home';
import NotFound from './pages/NotFound';
import Discover from './pages/Discover';
import Portfolio from './pages/Portfolio';
import Watchlist from './pages/Watchlist';
import SignIn from './pages/SignIn';
import Register from './pages/Register';
import { AuthProvider } from './context/AuthContext';
import IndividualCompany from './pages/IndividualCompany';
import TradeHistory from './pages/TradeHistory';
import Leaderboard from './pages/Leaderboard';
import RequireAuth from './context/RequireAuth';
import HasAuth from './context/HasAuth';

function App() {
  const location = useLocation();
  const hideRoutes = ['/signin', '/register'];

  const shouldHideRoutes = hideRoutes.includes(location.pathname);

  return (
    <AuthProvider>
      {!shouldHideRoutes && <NavBar />}
      <Routes>
        <Route
          path="/"
          element={
            <HasAuth>
              <Home />
            </HasAuth>
          }
        />
        <Route
          path="/discover"
          element={
            <RequireAuth>
              <Discover />
            </RequireAuth>
          }
        />
        <Route
          path="/portfolio"
          element={
            <RequireAuth>
              <Portfolio />
            </RequireAuth>
          }
        />
        <Route
          path="/watchlist"
          element={
            <RequireAuth>
              <Watchlist />
            </RequireAuth>
          }
        />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/register" element={<Register />} />
        <Route
          path="/history"
          element={
            <RequireAuth>
              <TradeHistory />
            </RequireAuth>
          }
        />
        <Route
          path="/leaderboard"
          element={
            <RequireAuth>
              <Leaderboard />
            </RequireAuth>
          }
        />
        <Route
          path="/stock/:stockId"
          element={
            <RequireAuth>
              <IndividualCompany />
            </RequireAuth>
          }
        />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </AuthProvider>
  );
}

function WrappedApp() {
  return (
    <BrowserRouter>
      <App />
    </BrowserRouter>
  );
}

export default WrappedApp;
