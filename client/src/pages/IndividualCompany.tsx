import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { useState } from 'react';
import appleLogo from '../assets/apple.png';
import CustomTooltip from '../components/CustomTooltip';

function IndividualCompany() {
  const dataHere = [
    {
      date: '2023-08-25',
      open: 147.28,
    },
    {
      date: '2023-08-24',
      open: 147.5,
    },
    {
      date: '2023-08-23',
      open: 146.35,
    },
    {
      date: '2023-08-22',
      open: 149.26,
    },
    {
      date: '2023-08-21',
      open: 149.65,
    },
    {
      date: '2023-08-18',
      open: 147.65,
    },
    {
      date: '2023-08-17',
      open: 150.88,
    },
  ];
  const [activeTab, setActiveTab] = useState('buy');

  return (
    <div className="lg:grid lg:grid-cols-4">
      <div className="lg:col-span-3">
        <div className="font-display text-white bg-main rounded-xl mt-5 mx-[16px] p-5 md:pb-12 md:mt-10">
          <div className="mb-10 md:ml-2 lg:ml-6">
            <h1 className="text-2xl font-bold mb-2 md:text-4xl">
              Market stats
            </h1>
          </div>
          <div className="md:flex md:flex-col md:justify-center">
            <div className="flex mb-7 md:self-center">
              <img
                className="h-10 w-10 md:h-20 md:w-20 self-center object-contain"
                alt="apple logo"
                src={appleLogo}
              />
              <span className="ml-2 self-center">
                <h2 className="text-lg font-bold md:text-3xl">Apple</h2>
                <p className="text-xs md:text-xl text-symbol">AAPL</p>
              </span>
            </div>
            <div className="md:self-center">
              <p className="text-neutraldark self-center px-8 md:px-2 md:mb-5 lg:ml-4 lg:text-lg md:hidden">
                Price{' '}
                <span className="text-white self-center float-right text-lg relative bottom-1 md:hidden">
                  $32,455.12
                </span>
              </p>
              <p
                className="md:text-2xl lg:text-3xl md:text-white md:block"
                hidden
              >
                $32,455.12
              </p>
            </div>
          </div>
          <ul className="grid grid-cols-2 w-full gap-4 mt-8 md:grid-cols-4">
            <li className="p-4 col-span-1 bg-background rounded-md">
              <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                Market Cap
              </h2>
              <p>$30,455.12</p>
            </li>
            <li className="p-4 col-span-1 md:row-span-1 bg-background rounded-md">
              <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                Previous Close
              </h2>
              <p>$30,455.12</p>
            </li>
            <li className="p-4 col-span-1 md:row-span-1 bg-background rounded-md">
              <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                52W Range
              </h2>
              <p>169-192</p>
            </li>
            <li className="p-4 col-span-1 md:row-span-1 bg-background rounded-md">
              <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                Stock Prediction
              </h2>
              <p>40% chance ^</p>
            </li>
          </ul>
        </div>
        <div className="flex font-display text-white bg-main rounded-xl mx-[16px] mt-5 p-5 pl-0">
          <ResponsiveContainer width="99%" height={350}>
            <AreaChart
              data={dataHere}
              margin={{ top: 0, right: 0, left: 0, bottom: 0 }}
            >
              <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#5266FE" stopOpacity={0.8} />
                  <stop offset="85%" stopColor="#5266FE" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" />
              <YAxis
                dataKey="open"
                domain={['dataMin-2', 'dataMax+2']}
                allowDecimals={false}
                tickLine={false}
                scale="sequential"
              />
              <Tooltip
                coordinate={{ y: 0 }}
                content={<CustomTooltip />}
                cursor={{ stroke: '#5266FE', strokeWidth: 1 }}
              />
              <Area
                dataKey="open"
                type="monotone"
                stroke="#5266FE"
                fill="url(#gradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="mt-5 bg-main rounded-xl lg:col-span-1 lg:mt-10 mx-[16px] lg:ml-0">
        <div className="flex flex-row font-display text-2xl md:text-3xl justify-center mt-5 pt-5 lg:pt-0">
          <button
            type="button"
            className={`${
              activeTab === 'buy'
                ? 'text-primary border-primary'
                : 'text-neutraldark'
            } border-b-4 px-7 md:px-8 2xl:px-16 pb-6`}
            onClick={() => setActiveTab('buy')}
          >
            Buy
          </button>
          <button
            type="button"
            className={`${
              activeTab === 'sell'
                ? 'text-primary border-primary'
                : 'text-neutraldark'
            } border-b-4 px-7 md:px-8 2xl:px-16 pb-6`}
            onClick={() => setActiveTab('sell')}
          >
            Sell
          </button>
        </div>
        {activeTab === 'buy' && (
          <div className="pb-10 lg:pb-0">
            <div className="flex justify-center mt-10">
              <img
                className="h-16 w-16 self-center object-contain"
                alt="apple logo"
                src={appleLogo}
              />
              <span className="ml-2">
                <h2 className="text-2xl font-bold text-white">BUY AAPL</h2>
                <p className="text-3xl font-bold text-symbol">179.91</p>
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
                Buy
              </button>
            </form>
          </div>
        )}
        {activeTab === 'sell' && (
          <div>
            <div className="flex justify-center mt-10">
              <img
                className="h-16 w-16 self-center object-contain"
                alt="apple logo"
                src={appleLogo}
              />
              <span className="ml-2">
                <h2 className="text-2xl font-bold text-white">SELL AAPL</h2>
                <p className="text-3xl font-bold text-symbol">179.91</p>
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
                Sell
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}

export default IndividualCompany;
