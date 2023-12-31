import { useContext, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AiOutlineStock,
  AiOutlineMenu,
  AiOutlineHistory,
} from 'react-icons/ai';
import { BsBookmark, BsBriefcase } from 'react-icons/bs';
import { MdLeaderboard } from 'react-icons/md';
import { AuthContext } from '../context/AuthContext';

function NavBar() {
  const [navbarOpen, setNavbarOpen] = useState(true);
  const { currentUser, signOut } = useContext(AuthContext);

  return (
    <nav>
      <header className="bg-main py-6 rounded-b-xl md:hidden">
        <h1 className="font-display text-lg font-bold text-white ml-6 hover:cursor-pointer">
          <Link to="/">TradePros</Link>
        </h1>
      </header>
      <button
        onClick={() => setNavbarOpen(!navbarOpen)}
        type="button"
        className="absolute right-9 top-[28px] text-white md:hidden"
      >
        <AiOutlineMenu size={20} />
      </button>
      <ul
        className="font-display mx-[16px] m-2 my-5 pt-5 pb-5 p-2 bg-main rounded-xl md:flex md:space-y-0 md:m-0 md:rounded-none md:py-6 md:text-base md:p-0 md:items-center md:justify-between md:whitespace-nowrap"
        hidden={navbarOpen}
      >
        <li
          className="navbar-item ml-6 md:flex text-2xl md:text-white md:hover:text-white font-bold"
          hidden
        >
          <Link to="/">TradePros</Link>
        </li>
        {currentUser && (
          <div className="flex flex-col md:flex-row">
            <li className="navbar-item ml-6 mb-6 md:mb-0 md:hover:underline md:hover:underline-offset-[33px]">
              <div className="flex flex-row items-center">
                <Link className="text-white navbar-item flex" to="/discover">
                  <AiOutlineStock
                    size={20}
                    className="md:hidden mr-2 self-center"
                  />
                  Discover
                </Link>
              </div>
            </li>
            <li className="navbar-item ml-6 mb-6 md:mb-0 md:mx-6 lg:mx-14 md:hover:underline md:hover:underline-offset-[33px]">
              <div className="flex flex-row items-center">
                <Link className="text-white navbar-item flex" to="/watchlist">
                  <BsBookmark
                    size={16}
                    className="md:hidden mr-2 self-center"
                  />
                  Watchlist
                </Link>
              </div>
            </li>
            <li className="mb-6 md:mb-0 navbar-item ml-6 md:ml-0 md:hover:underline md:hover:underline-offset-[33px]">
              <div className="flex flex-row items-center">
                <Link className="text-white navbar-item flex" to="/portfolio">
                  <BsBriefcase
                    size={16}
                    className="md:hidden mr-2 self-center"
                  />
                  Portfolio
                </Link>
              </div>
            </li>
            <li className="navbar-item ml-6 mb-6 md:mb-0 md:mx-6 lg:mx-14 md:hover:underline md:hover:underline-offset-[33px]">
              <div className="flex flex-row items-center">
                <Link className="text-white navbar-item flex" to="/history">
                  <AiOutlineHistory
                    size={16}
                    className="md:hidden mr-2 self-center"
                  />
                  Trade History
                </Link>
              </div>
            </li>
            <li className="mb-8 md:mb-0 navbar-item mx-6 md:ml-0 md:hover:underline md:hover:underline-offset-[33px]">
              <div className="flex items-center">
                <Link className="text-white navbar-item flex" to="/leaderboard">
                  <MdLeaderboard
                    size={16}
                    className="md:hidden mr-2 self-center"
                  />
                  <span>Leaderboard</span>
                </Link>
              </div>
            </li>
          </div>
        )}
        {!currentUser ? (
          <div className="flex justify-evenly mx-2 mr-6">
            <li className="navbar-item mr-2 w-full md:ml-0 md:mr-3 p-1 md:mb-0 rounded-lg bg-signinmobile md:bg-main text-center text-lg font-bold md:px-4">
              <Link
                className="text-primarydark md:font-normal md:font-display"
                to="/signin"
              >
                Sign In
              </Link>
            </li>
            <li className="navbar-item ml-2 w-full md:ml-0 md:mr-3 p-1 rounded-lg bg-primarydark text-center text-lg font-bold md:px-4">
              <Link
                className="text-white md:font-normal md:font-display"
                to="/register"
              >
                Register
              </Link>
            </li>
          </div>
        ) : (
          <div className="flex justify-evenly mr-6">
            <li className="navbar-item w-full md:ml-0 md:mr-3 p-1 rounded-lg bg-primarydark text-center text-base font-bold md:px-2">
              <button
                type="button"
                className="text-white md:font-normal md:font-display"
                onClick={(e) => {
                  e.preventDefault();
                  signOut();
                }}
              >
                Sign Out
              </button>
            </li>
          </div>
        )}
      </ul>
    </nav>
  );
}

export default NavBar;
