import { PulseLoader } from 'react-spinners';

type BuySellFormTypes = {
  type: string;
  logo: string;
  ticker: string;
  price: number;
  loading: boolean;
};

function BuySellForm({ type, logo, ticker, price, loading }: BuySellFormTypes) {
  return (
    <div className="pb-10 lg:pb-0 md:mt-0 lg:mt-24">
      <div className="flex justify-center mt-10">
        {!loading ? (
          <img
            className="h-16 w-16 self-center object-contain text-white font-display text-sm"
            alt="company logo"
            src={logo}
          />
        ) : (
          <div className="flex justify-center h-16 w-16">
            <div className="self-center">
              <PulseLoader
                loading={loading}
                size={5}
                aria-label="Loading Spinner"
                color="#5266FE"
              />
            </div>
          </div>
        )}
        <span className="ml-2">
          <h2 className="text-2xl font-bold text-white">
            {type === 'buy' ? 'BUY' : 'SELL'} {ticker}
          </h2>
          <p className="text-3xl font-bold text-symbol">${price.toFixed(2)}</p>
        </span>
      </div>
      <div className="flex flex-row justify-around text-white mt-10 font-display text-2xl">
        <p>100,000</p>
        <p>0</p>
      </div>
      <form className="flex flex-col mt-16 px-5 md:px-20 lg:px-5 gap-5">
        <h2 className="font-bold text-white text-center">Amount</h2>
        <input
          type="text"
          placeholder="Amount"
          className="bg-background rounded-md text-white caret-primary focus:outline-none focus:border focus:border-primary pl-2 py-3 md:py-4 md:pl-3"
        />
        <h2 className="font-bold text-white text-center mt-4">Unit</h2>
        <input
          type="text"
          placeholder="Unit"
          className="bg-background rounded-md text-white caret-primary focus:outline-none focus:border focus:border-primary pl-2 py-3 md:py-4 md:pl-3"
        />
        <button
          type="button"
          className="rounded-2xl bg-primary font-display font-bold text-white p-3 text-lg md:text-2xl mt-12 md:mt-16 lg:mt-28"
        >
          {type === 'buy' ? 'BUY' : 'SELL'}
        </button>
      </form>
    </div>
  );
}

export default BuySellForm;
