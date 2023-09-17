import { SyntheticEvent, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { signIn } from '../firebase/firebase';
import { errorMap } from '../constants/config';

function SignIn() {
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const location = useLocation();
  const [err, setErr] = useState('');
  const navigate = useNavigate();

  const login = (e: SyntheticEvent) => {
    e.preventDefault();
    signIn(email, password)
      .then(() => {
        const from = location.state?.from?.pathname || '/portfolio';
        navigate(from, { replace: true });
      })
      .catch((response) => {
        setErr(errorMap[response.message]);
      });
  };
  return (
    <>
      <ul className="text-white font-display">
        <li className="absolute top-10 left-10 text-2xl select-none md:text-3xl md:left-28 font-bold">
          TradePros
        </li>
        <li className="absolute right-10 top-[43px] text-lg text-primarydark md:text-xl md:right-28 md:top-[45px]">
          <Link to="/">Back</Link>
        </li>
      </ul>
      <div className="flex items-center justify-center h-screen">
        <div className="font-display text-white w-full md:w-3/4 lg:w-6/12">
          <form
            onSubmit={login}
            className="flex flex-col bg-main mx-5 rounded-xl lg:rounded-none"
          >
            <h2 className="my-5 mb-5 mx-5 font-bold text-lg text-center select-none md:text-2xl">
              Sign In
            </h2>
            {err ? (
              <div className="text-xl font-display text-warning justify-center items-center">
                <p className="text-center self-center">{err}</p>
              </div>
            ) : null}
            <div className="flex flex-col gap-3 px-5 md:gap-8 mt-5">
              <input
                type="text"
                placeholder="Email"
                className="bg-background rounded-md text-white caret-primarydark focus:outline-none focus:border focus:border-primarydark pl-2 py-2 md:py-3 md:pl-3"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <input
                type="password"
                placeholder="Password"
                className="bg-background rounded-md text-white caret-primarydark focus:outline-none focus:border focus:border-primarydark pl-2 py-2 md:py-3 md:pl-3"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <div className="flex justify-end mr-5 text-sm text-primarydark my-4">
              <button type="button">Forgot Password?</button>
            </div>
            <button
              type="submit"
              className="bg-primarydark rounded-md px-10 py-2 mx-5 mb-5 lg:mt-7"
            >
              Sign In
            </button>
            <div className="mb-5 lg:my-5">
              <div className="flex justify-center mb-1">
                <p>If you don&apos;t have an account</p>
              </div>
              <div className="flex justify-center text-sm text-primarydark">
                <Link to="/register">Register Here!</Link>
              </div>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}

export default SignIn;
