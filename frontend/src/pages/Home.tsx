import Footer from './Footer';

function Home() {
  return (
    <div>
      <div className="flex flex-col bg-[#121318] mt-5 mx-[16px] md:mx-6 lg:mx-8 rounded-xl md:pt-0 md:pb-5 md:mt-10 lg:pb-[220px] md:flex-row">
        <div className="md:mr-20 lg:mr-20">
          <p className="py-[132px] rounded-xl m-2 bg-black md:hidden">asdas</p>
          <section className="font-display text-[#9395A5] md:pt-0 lg:mt-40">
            {/* Note: Image is around 337px vertically */}
            <div className="ml-[23px] md:mt-20 md:ml-20">
              <h2 className="mb-7 font-display font-bold md:text-xl lg:text-4xl text-[#5367FE]">
                Buy and Sell
              </h2>
              <p className="text-xs md:max-w-3xl lg:max-w-xl">
                Dive into the stock market world without real money at risk.
                Invest in top companies like Apple, Google, and more using
                virtual funds and build your trading skills.
              </p>
            </div>
          </section>
          <button
            type="button"
            className="bg-[#5367FE] text-white rounded-md p-2 mx-[20px] mt-[42px] mb-4 md:ml-20"
          >
            Start Investing
          </button>
        </div>
        <div
          className="md:flex bg-black rounded-xl md:mt-10 lg:mt-20 md:mr-20 lg:pb-96"
          hidden
        >
          <p>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur
            et nisi vel turpis fermentum posuere. Suspendisse laoreet tristique
            laoreet. Duis egestas tincidunt ligula ut sollicitudin.
          </p>
        </div>
      </div>
      <div className="bg-[#121318] mt-16 md:flex md:flex-row md:py-5 md:pl-7">
        <div className="text-white flex flex-col text-center font-display pt-24 md:justify-center md:max-w-sm md:pt-0 lg:max-w-3xl lg:pl-10 md:pr-5">
          <h3 className="text-[#5367FE] text-md lg:text-lg mb-2 md:text-left">
            Create Profile
          </h3>
          <h4 className=" text-2xl lg:text-3xl mb-5 md:text-left">
            Easy Way to Get Started
          </h4>
          <p className="text-[#9395A5] p-5 pt-0 text-center text-sm lg:text-base md:text-left md:p-0">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur
            et nisi vel turpis fermentum posuere. Suspendisse laoreet tristique
            laoreet. Duis egestas tincidunt ligula ut sollicitudin.
          </p>
        </div>
        <div className="text-white flex flex-col items-center font-display md:mr-5">
          <ul className="md:columns-2">
            <li className="rounded-lg bg-[#080808] m-2 p-5 mb-4 pl-8">
              <h5 className="font-bold text-md mb-2">
                <span className="text-[#5367FE]">1.</span> Create An Account
              </h5>
              <p className="text-[#9395A5] text-sm">
                Sign up with your email in under 2 minutes
              </p>
            </li>
            <li className="rounded-lg bg-[#080808] m-2 p-5 mb-4 pl-8">
              <h5 className="font-bold text-md mb-2">
                <span className="text-[#5367FE]">2.</span> Analyse Companies
              </h5>
              <p className="text-[#9395A5] text-sm">
                Determine which companies you want to invest into
              </p>
            </li>
            <li className="rounded-lg bg-[#080808] m-2 p-5 mb-4 pl-8">
              <h5 className="font-bold text-md mb-2">
                <span className="text-[#5367FE]">3.</span> Start Investing
                Instantly
              </h5>
              <p className="text-[#9395A5] text-sm">
                Buy and Sell a variety of stocks with a $100,000 of virtual
                money
              </p>
            </li>
            <li className="rounded-lg bg-[#080808] m-2 p-5 mb-4 pl-8">
              <h5 className="font-bold text-md mb-2">
                <span className="text-[#5367FE]">4.</span> Compete Against Your
                Friends
              </h5>
              <p className="text-[#9395A5] text-sm">
                See who has profited the most using our leaderboard
              </p>
            </li>
          </ul>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default Home;
