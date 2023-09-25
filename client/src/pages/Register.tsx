import { SyntheticEvent, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { registerUser } from '../firebase/firebase';

function Register() {
  const [username, setUsername] = useState<string>('');
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [err, setErr] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();

  const register = (e: SyntheticEvent) => {
    e.preventDefault();

    axios
      .post('/register', {
        username,
        email,
        password,
        confirmPassword,
      })
      .then((response) => {
        registerUser(response.data).then(() => {
          const from = location.state?.from?.pathname || '/portfolio';
          navigate(from, { replace: true });
        });
      })
      .catch(({ response }) => {
        setErr(response.data.message);
      });
  };

  return (
    <>
      <ul className="text-white font-display">
        <li className="absolute top-10 left-10 text-2xl select-none md:text-3xl md:left-28 font-bold ">
          TradePros
        </li>
        <li className="absolute right-10 top-[43px] text-lg text-primarydark md:text-xl md:right-28 md:top-[45px]">
          <Link to="/">Back</Link>
        </li>
      </ul>
      <div className="flex items-center justify-center h-screen">
        <div className="font-display text-white w-full md:w-3/4 lg:w-6/12">
          <form
            onSubmit={register}
            className="flex flex-col bg-main mx-5 rounded-xl lg:rounded-none"
          >
            <h2 className="my-5 mb-5 mx-5 font-bold text-lg text-center select-none md:text-2xl">
              Register
            </h2>
            {err ? (
              <div className="text-sm md:text-xl font-display text-warning justify-center items-center">
                <p className="text-center self-center">{err}</p>
              </div>
            ) : null}
            <div className="flex flex-col gap-4 px-5 md:gap-4 mb-7 mt-5">
              <input
                type="text"
                placeholder="Username"
                className="bg-background rounded-md text-white caret-primarydark focus:outline-none focus:border focus:border-primarydark pl-2 py-2 md:py-3 md:pl-3"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
              <input
                type="email"
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
              <input
                type="password"
                placeholder="Confirm Password"
                className="bg-background rounded-md text-white caret-primarydark focus:outline-none focus:border focus:border-primarydark pl-2 py-2 md:py-3 md:pl-3"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </div>
            <button
              type="submit"
              className="bg-primary rounded-md px-10 py-2 mx-5 mb-5 lg:mt-3"
            >
              Register
            </button>
            <div className="mb-5 lg:my-5">
              <div className="flex justify-center mb-1">
                <p>Already have an account?</p>
              </div>
              <div className="flex justify-center text-sm text-primarydark">
                <Link to="/signin">Log In</Link>
              </div>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}

export default Register;
