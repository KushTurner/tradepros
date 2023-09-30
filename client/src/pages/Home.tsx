import { useNavigate } from 'react-router-dom';
import { useMediaQuery } from 'react-responsive';
import Instructions from '../components/Instructions';
import Footer from './Footer';
import { instructionsData } from '../constants/config';

function Home() {
  const navigate = useNavigate();
  const sm = useMediaQuery({ query: '(min-width: 600px)' });
  const md = useMediaQuery({ query: '(min-width: 768px)' });
  const lg = useMediaQuery({ query: '(min-width: 1024px)' });
  const xl = useMediaQuery({ query: '(min-width: 1024px)' });
  const xxl = useMediaQuery({ query: '(min-width: 1536px)' });

  const res = () => {
    if (xxl) {
      return 720;
    }
    if (xl) {
      return 600;
    }
    if (lg) {
      return 450;
    }
    if (md) {
      return 400;
    }
    if (sm) {
      return 500;
    }
    return 340;
  };

  const iframeRes = res();

  return (
    <div>
      <div className="flex flex-col-reverse md:grid grid-cols-12 items-center bg-main mt-5 mx-[16px] md:mx-6 lg:mx-8 rounded-xl md:pt-[25px] md:pb-[50px]  lg:pt-[50px] lg:pb-[100px] md:py md:mt-10">
        <div className="md:col-span-4 flex justify-center pb-10 md:pb-0">
          <section className="font-display text-neutral flex flex-col px-5 md:px-0 self-center mt-12 md:mt-5 lg:mt-0 md:ml-8 lg:ml-10 xl:ml-20">
            <div>
              <h2 className="mb-7 col-span- font-display font-bold md:text-xl lg:text-4xl text-primary">
                Buy and Sell
              </h2>
              <p className="text-sm lg:text-base md:w-full lg:w-full">
                Dive into the stock market world without real money at risk.
                Invest in top companies like Apple, Google, and more using
                virtual funds and build your trading skills.
              </p>
            </div>
            <button
              type="button"
              className="bg-primarydark text-white rounded-md md:w-3/4 mt-10 p-2"
              onClick={() => {
                navigate('/register');
              }}
            >
              Start Investing
            </button>
          </section>
        </div>
        <div className="col-span-8 flex mt-8 md:mt-4 justify-center">
          <iframe
            src="https://www.youtube.com/embed/s-4DVAUDIqA?si=Dgu3TQ_-BIyUfM9I&rel=0&fs=0&controls=0"
            title="Tradepros"
            width={`${iframeRes}`}
            height={`${(iframeRes * 9) / 16}`}
            allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          />
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
            Whether you&apos;re a seasoned investor or just getting started, our
            platform offers a risk-free way to hone your skills and grow your
            financial portfolio.
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
