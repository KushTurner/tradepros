/* eslint-disable react/require-default-props */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { User } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import { createContext, useState, useEffect, ReactNode } from 'react';
import { SignOutUser, userStateListener } from '../firebase/firebase';

interface Props {
  children?: ReactNode;
}

export const AuthContext = createContext({
  currentUser: {} as User | null,
  loading: true,
  setCurrentUser: (_user: User) => {},
  signOut: () => {},
});

export function AuthProvider({ children }: Props) {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = userStateListener((user) => {
      if (user) {
        setCurrentUser(user);
      }
      setLoading(false);
    });
    return unsubscribe;
  }, [setCurrentUser]);

  const signOut = () => {
    SignOutUser();
    setCurrentUser(null);
    navigate('/');
  };

  // eslint-disable-next-line react/jsx-no-constructed-context-values
  const value = {
    currentUser,
    loading,
    setCurrentUser,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
