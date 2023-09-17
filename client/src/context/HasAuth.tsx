import { useContext } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { AuthContext } from './AuthContext';

function HasAuth({ children }: { children: JSX.Element }) {
  const { currentUser, loading } = useContext(AuthContext);
  const location = useLocation();

  if (loading) {
    return null;
  }

  if (currentUser && location.pathname === '/') {
    return <Navigate to="/portfolio" />;
  }

  return children;
}

export default HasAuth;
