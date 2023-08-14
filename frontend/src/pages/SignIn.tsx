import { Link } from 'react-router-dom';

function SignIn() {
  return (
    <>
      <ul className="text-white font-display">
        <li className="absolute top-10 left-10 text-2xl select-none md:text-3xl md:left-28 font-bold ">
          TradePros
        </li>
        <li className="absolute right-10 top-[43px] text-lg text-[#5266FE] md:text-xl md:right-28 md:top-[45px]">
          <Link to="/">Back</Link>
        </li>
      </ul>
      <div className="flex items-center justify-center h-screen">
        <div className="font-display text-white w-full md:w-3/4 lg:w-6/12">
          <form className="flex flex-col bg-[#121318] mx-5 rounded-xl lg:rounded-none">
            <h2 className="my-5 mb-10 ml-5 font-bold text-lg text-center select-none md:text-2xl">
              Sign In
            </h2>
            <div className="flex flex-col gap-3 px-5 md:gap-8">
              <input
                type="email"
                placeholder="Email"
                className="bg-[#080808] rounded-md text-white caret-[#5266FE] focus:outline-none focus:border focus:border-[#5266FE] pl-2 py-2 md:py-3 md:pl-3"
              />
              <input
                type="password"
                placeholder="Password"
                className="bg-[#080808] rounded-md text-white caret-[#5266FE] focus:outline-none focus:border focus:border-[#5266FE] pl-2 py-2 md:py-3 md:pl-3"
              />
            </div>
            <div className="flex justify-end mr-5 text-sm text-[#5266FE] my-4">
              <button type="button">Forgot Password?</button>
            </div>
            <button
              type="submit"
              className="bg-[#5266FE] rounded-md px-10 py-2 mx-5 mb-5 lg:mt-7"
            >
              Sign In
            </button>
            <div className="mb-5 lg:my-5">
              <div className="flex justify-center mb-1">
                <p>If you don&apos;t have an account</p>
              </div>
              <div className="flex justify-center text-sm text-[#5266FE]">
                <button type="button">Register Here!</button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}

export default SignIn;
