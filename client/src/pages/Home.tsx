import Instructions from '../components/Instructions';
import Footer from './Footer';

const instructionsData = [
  {
    index: 1,
    header: 'Create An Account',
    description: 'Sign up with your email in under 2 minutes',
  },
  {
    index: 2,
    header: 'Analyse Companies',
    description: 'Determine which companies you want to invest into',
  },
  {
    index: 3,
    header: 'Start Investing Instantly',
    description:
      'Buy and Sell a variety of stocks with a $100,000 of virtual money',
  },
  {
    index: 4,
    header: 'Compete Against Your Friends',
    description: 'See who has profited the most using our leaderboard',
  },
];

function Home() {
  return (
    <div>
      <div className="flex flex-col bg-main mt-5 mx-[16px] md:mx-6 lg:mx-8 rounded-xl md:pt-0 md:pb-5 md:mt-10 lg:pb-[220px] md:flex-row">
        <div className="md:mr-20 lg:mr-20">
          <p className="py-[132px] rounded-xl m-2 bg-black md:hidden">asdas</p>
          <section className="font-display text-neutral md:pt-0 lg:mt-40">
            {/* Note: Image is around 337px vertically */}
            <div className="ml-[23px] md:mt-20 md:ml-20">
              <h2 className="mb-7 font-display font-bold md:text-xl lg:text-4xl text-primary">
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
            className="bg-primarydark text-white rounded-md p-2 mx-[20px] mt-[42px] mb-4 md:ml-20"
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
      <div className="bg-main mt-16 md:flex md:flex-row md:py-5 md:pl-7">
        <div className="text-white flex flex-col text-center font-display pt-24 md:justify-center md:max-w-sm md:pt-0 lg:max-w-3xl lg:pl-10 md:pr-5">
          <h3 className="text-primary text-md lg:text-lg mb-2 md:text-left">
            Create Profile
          </h3>
          <h4 className=" text-2xl lg:text-3xl mb-5 md:text-left">
            Easy Way to Get Started
          </h4>
          <p className="text-neutral p-5 pt-0 text-center text-sm lg:text-base md:text-left md:p-0">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur
            et nisi vel turpis fermentum posuere. Suspendisse laoreet tristique
            laoreet. Duis egestas tincidunt ligula ut sollicitudin.
          </p>
        </div>
        <div className="text-white flex flex-col items-center font-display md:mr-5">
          <ul className="grid grid-cols-1 md:grid-cols-2">
            {instructionsData.map((instructs) => (
              <Instructions
                key={instructs.index}
                index={instructs.index}
                header={instructs.header}
                description={instructs.description}
              />
            ))}
          </ul>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default Home;
