/* eslint-disable react/function-component-definition */
/* eslint-disable react/prop-types */

import { createContext, useContext, useMemo, useState } from 'react';

interface AuthData {
  token: string;
  loggedIn: boolean;
}

interface AuthContextType {
  authData: AuthData;
  login: (token: string) => void;
  logout: () => void;
}

interface AuthProviderProps {
  children: React.ReactNode;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [authData, setAuthData] = useState<AuthData>({
    token: '',
    loggedIn: false,
  });

  const login = (token: string) => {
    setAuthData({
      token,
      loggedIn: true,
    });
  };

  const logout = () => {
    setAuthData({
      token: '',
      loggedIn: false,
    });
  };

  const memoizedValue = useMemo(() => {
    return {
      authData,
      login,
      logout,
    };
  }, [authData]);

  return (
    <AuthContext.Provider value={memoizedValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined || context.authData === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
