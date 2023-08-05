import { useState } from 'react';
import { Link } from 'react-router-dom';
import { AiOutlineStock, AiOutlineMenu } from 'react-icons/ai';
import { BsBookmark, BsBriefcase } from 'react-icons/bs';

function NavBar() {
  const [navbarOpen, setNavbarOpen] = useState(true);

  return (
    <nav>
      <header className="bg-[#121318] py-6 rounded-b-xl md:hidden">
        <h1 className="font-display text-lg font-bold text-white ml-6">
          TradePros
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
        className="font-display mx-[16px] m-2 my-5 pt-5 pb-5 p-2 bg-[#121318] rounded-xl md:flex md:space-y-0 md:m-0 md:rounded-none md:py-6 md:text-base md:p-0 md:items-center md:justify-between"
        hidden={navbarOpen}
      >
        <li
          className="navbar-item ml-6 md:flex text-2xl md:text-white md:hover:text-white font-bold"
          hidden
        >
          <Link to="/">TradePros</Link>
        </li>
        <div className="flex flex-col md:flex-row">
          <li className="navbar-item ml-6 mb-6 md:mb-0 md:hover:underline md:hover:underline-offset-[36px]">
            <div className="flex flex-row items-center">
              <AiOutlineStock size={20} className="md:hidden mr-2" />
              <Link className="text-white" to="/market">
                Market
              </Link>
            </div>
          </li>
          <li className="navbar-item ml-6 mb-6 md:mb-0 md:mx-6 lg:mx-20 md:hover:underline md:hover:underline-offset-[33px]">
            <div className="flex flex-row items-center">
              <BsBookmark size={16} className="md:hidden mr-2" />
              <Link className="text-white" to="/watchlist">
                Watchlist
              </Link>
            </div>
          </li>
          <li className="navbar-item ml-6 md:ml-0 md:hover:underline md:hover:underline-offset-[33px]">
            <div className="flex flex-row items-center">
              <BsBriefcase size={16} className="md:hidden mr-2" />
              <Link className="text-white" to="/portfolio">
                Portfolio
              </Link>
            </div>
          </li>
        </div>
        <div className="mt-8 flex justify-evenly mx-2 mr-6">
          <li className="navbar-item mr-2 w-full md:ml-0 md:mr-3 p-1 md:mb-0 rounded-lg bg-[#E8E9ED] md:bg-[#121318] text-center text-lg font-bold md:px-4">
            <button type="button" className="text-[#5266FE] md:font-normal">
              Sign in
            </button>
          </li>
          <li className="navbar-item ml-2 w-full md:ml-0 md:mr-3 p-1 rounded-lg bg-[#5266FE] text-center text-lg font-bold md:px-4">
            <button type="button" className="text-[#E8E9ED] md:font-normal">
              Register
            </button>
          </li>
        </div>
      </ul>
    </nav>
  );
}

export default NavBar;
