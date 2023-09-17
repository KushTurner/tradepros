import { initializeApp } from 'firebase/app';
import {
  NextOrObserver,
  User,
  getAuth,
  onAuthStateChanged,
  signInWithCustomToken,
  signInWithEmailAndPassword,
  signOut,
} from 'firebase/auth';
import getFirebaseConfig from './firebase-config';

const app = initializeApp(getFirebaseConfig());
const auth = getAuth(app);

export const signIn = (email: string, password: string) => {
  return signInWithEmailAndPassword(auth, email, password);
};

export const registerUser = (token: string) => {
  return signInWithCustomToken(auth, token);
};

export const userStateListener = (callback: NextOrObserver<User>) => {
  return onAuthStateChanged(auth, callback);
};

export const SignOutUser = async () => signOut(auth);
