import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import NavBar from './components/NavBar';
import Home from './pages/Home';
import NotFound from './pages/NotFound';
import Market from './pages/Market';
import Portfolio from './pages/Portfolio';
import Watchlist from './pages/Watchlist';
import SignIn from './pages/SignIn';
import Register from './pages/Register';
import { AuthProvider } from './context/AuthContext';
import IndividualCompany from './pages/IndividualCompany';
import TradeHistory from './pages/TradeHistory';
import Leaderboard from './pages/Leaderboard';

function App() {
  const location = useLocation();
  const hideRoutes = ['/signin', '/register'];

  const shouldHideRoutes = hideRoutes.includes(location.pathname);
  return (
    <AuthProvider>
      {!shouldHideRoutes && <NavBar />}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/market" element={<Market />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/watchlist" element={<Watchlist />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/register" element={<Register />} />
        <Route path="/history" element={<TradeHistory />} />
        <Route path="/leaderboard" element={<Leaderboard />} />
        <Route path="/stock/:stockId" element={<IndividualCompany />} />
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
