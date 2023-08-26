import { useState } from 'react';
import { Link } from 'react-router-dom';
import { AiOutlineStock, AiOutlineMenu } from 'react-icons/ai';
import { BsBookmark, BsBriefcase } from 'react-icons/bs';
import { useAuth } from '../context/AuthContext';

function NavBar() {
  const [navbarOpen, setNavbarOpen] = useState(true);
  const { authData, logout } = useAuth();

  return (
    <nav>
      <header className="bg-[#121318] py-6 rounded-b-xl md:hidden">
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
        className="font-display mx-[16px] m-2 my-5 pt-5 pb-5 p-2 bg-[#121318] rounded-xl md:flex md:space-y-0 md:m-0 md:rounded-none md:py-6 md:text-base md:p-0 md:items-center md:justify-between md:whitespace-nowrap"
        hidden={navbarOpen}
      >
        <li
          className="navbar-item ml-6 md:flex text-2xl md:text-white md:hover:text-white font-bold"
          hidden
        >
          <Link to="/">TradePros</Link>
        </li>
        {authData.loggedIn && (
          <div className="flex flex-col md:flex-row">
            <li className="navbar-item ml-6 mb-6 md:mb-0 md:hover:underline md:hover:underline-offset-[36px]">
              <div className="flex flex-row items-center">
                <Link className="text-white navbar-item flex" to="/market">
                  <AiOutlineStock
                    size={20}
                    className="md:hidden mr-2 self-center"
                  />
                  Market
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
                <Link className="text-white navbar-item flex" to="/watchlist">
                  <BsBookmark
                    size={16}
                    className="md:hidden mr-2 self-center"
                  />
                  Trade History
                </Link>
              </div>
            </li>
            <li className="mb-8 md:mb-0 navbar-item ml-6 md:ml-0 md:hover:underline md:hover:underline-offset-[33px]">
              <div className="flex items-center">
                <Link className="text-white navbar-item flex" to="/portfolio">
                  <BsBriefcase
                    size={16}
                    className="md:hidden mr-2 self-center"
                  />
                  <span>Leaderboard</span>
                </Link>
              </div>
            </li>
          </div>
        )}
        {!authData.loggedIn ? (
          <div className="flex justify-evenly mx-2 mr-6">
            <li className="navbar-item mr-2 w-full md:ml-0 md:mr-3 p-1 md:mb-0 rounded-lg bg-[#E8E9ED] md:bg-[#121318] text-center text-lg font-bold md:px-4">
              <Link
                className="text-[#5266FE] md:font-normal md:font-display"
                to="/signin"
              >
                Sign In
              </Link>
            </li>
            <li className="navbar-item ml-2 w-full md:ml-0 md:mr-3 p-1 rounded-lg bg-[#5266FE] text-center text-lg font-bold md:px-4">
              <Link
                className="text-white md:font-normal md:font-display"
                to="/register"
              >
                Register
              </Link>
            </li>
          </div>
        ) : (
          <div className="flex justify-evenly mx-2 mr-6">
            <li className="navbar-item ml-2 w-full md:ml-0 md:mr-3 p-1 rounded-lg bg-[#5266FE] text-center text-lg font-bold md:px-4">
              <button
                type="button"
                className="text-white md:font-normal md:font-display"
                onClick={(e) => {
                  e.preventDefault();
                  logout();
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
