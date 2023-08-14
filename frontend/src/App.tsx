import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom';
import NavBar from './components/NavBar';
import Home from './pages/Home';
import NotFound from './pages/NotFound';
import Market from './pages/Market';
import Portfolio from './pages/Portfolio';
import Watchlist from './pages/Watchlist';
import SignIn from './pages/SignIn';
import Register from './pages/Register';

function App() {
  const location = useLocation();
  const hideRoutes = ['/signin', '/register'];

  const shouldHideRoutes = hideRoutes.includes(location.pathname);
  return (
    <>
      {!shouldHideRoutes && <NavBar />}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/market" element={<Market />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/watchlist" element={<Watchlist />} />
        <Route path="/signin" element={<SignIn />} />
        <Route path="/register" element={<Register />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
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
