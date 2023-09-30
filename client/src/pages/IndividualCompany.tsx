/* eslint-disable react-hooks/exhaustive-deps */
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { useContext, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { PulseLoader } from 'react-spinners';
import {
  AiFillStar,
  AiOutlineArrowDown,
  AiOutlineArrowUp,
  AiOutlineInfoCircle,
  AiOutlineStar,
} from 'react-icons/ai';
import axios from 'axios';
import CustomTooltip from '../components/CustomTooltip';
import {
  StockObject,
  StockPredictionObject,
  chartConfig,
} from '../constants/config';
import {
  convertDateToUnixTimestamp,
  convertUnixTimestampToDateTime,
  createDate,
} from '../helpers/date-helper';
import { HistoricalData, DataObject } from '../constants/types';
import ChartFilter from '../components/ChartFilter';
import BuySellForm from '../components/BuySellForm';
import formatMarketCap from '../helpers/money-helper';
import Footer from './Footer';
import { AuthContext } from '../context/AuthContext';

function IndividualCompany() {
  const navigate = useNavigate();
  const { currentUser } = useContext(AuthContext);
  const { stockId } = useParams();
  const [filter, setFilter] = useState('1D');
  const [activeTab, setActiveTab] = useState<'buy' | 'sell'>('buy');
  const [watchlist, setWatchlist] = useState(false);
  const [closedMarket, setClosedMarket] = useState(false);
  const [popUpHidden, setPopUpHidden] = useState(true);
  const [historicalData, setHistoricalData] = useState<DataObject>({});
  const [loadingHistoricalData, setLoadingHistoricalData] = useState(false);
  const [stockData, setStockData] = useState<StockObject>();
  const [stockPrediction, setStockPrediction] =
    useState<StockPredictionObject>();

  const formatData = (histData: HistoricalData) => {
    return histData.c.map((item: number, index: number) => {
      return {
        value: Number(item.toFixed(2)),
        date: convertUnixTimestampToDateTime(histData.t[index], filter),
      };
    });
  };

  const toggleStar = () => {
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      if (watchlist) {
        axios.delete(`/watchlist/${stockId}`, config);
      } else {
        const formData = new FormData();
        if (!stockId) return;
        const postConfig = {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'multipart/form-data',
          },
        };
        formData.append('stock', stockId);
        axios.post('/watchlist', formData, postConfig);
      }
      setWatchlist(!watchlist);
    });
  };

  useEffect(() => {
    if (!stockId) return;

    axios
      .get(`/stock?symbol=${stockId.toUpperCase()}`)
      .then(({ data }) => {
        setStockData({
          companyName: data.name,
          ticker: data.ticker,
          logo: data.logo,
          marketCap: data.marketCapitalization,
          price: data.c,
          previousClose: data.pc,
          weekHigh: data.metric['52WeekHigh'],
          weekLow: data.metric['52WeekLow'],
        });
      })
      .catch(() => {
        navigate('/discover');
      });
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      axios
        .get(`/watchlist/${stockId.toUpperCase()}`, config)
        .then(({ data }) => {
          if (data === true) {
            setWatchlist(!watchlist);
          }
        });
    });
    setLoadingHistoricalData(true);
    axios
      .get(`stock/prediction?symbol=${stockId.toUpperCase()}`)
      .then(({ data }) => {
        setStockPrediction({
          confidence: data.confidence,
          direction: data.model_answer,
        });
      });
    setLoadingHistoricalData(false);
  }, [stockId]);

  useEffect(() => {
    if (!stockId) return;
    if (historicalData[filter]) return;

    const getDateRange = () => {
      const { days, weeks, months, years } = chartConfig[filter];

      const endDate = new Date();
      const startDate = createDate(endDate, -days, -weeks, -months, -years);

      const startTimestampUnix = convertDateToUnixTimestamp(startDate);
      const endTimestampUnix = convertDateToUnixTimestamp(endDate);

      return { startTimestampUnix, endTimestampUnix };
    };

    const { startTimestampUnix, endTimestampUnix } = getDateRange();
    const res = chartConfig[filter].resolution;
    setLoadingHistoricalData(true);
    axios
      .get(
        `/stock/candle?symbol=${stockId.toUpperCase()}&resolution=${res}&from=${startTimestampUnix}&to=${endTimestampUnix}`
      )
      .then(({ data }) => {
        if (data.s === 'no_data') {
          setClosedMarket(!closedMarket);
          setFilter('1W');
        } else {
          setHistoricalData((prevData) => ({
            ...prevData,
            [filter]: formatData(data),
          }));
        }
        setLoadingHistoricalData(false);
      });
  }, [filter]);

  return (
    <div>
      <div className="lg:grid lg:grid-cols-4">
        <div className="lg:col-span-3">
          <div className="font-display text-white bg-main rounded-xl mt-5 mx-[16px] p-5 md:pb-12 md:mt-10">
            <div className="mb-10 md:ml-2 lg:ml-6 flex flex-row justify-between">
              <h1 className="text-2xl font-bold mb-2 md:text-4xl">
                Market stats
              </h1>
              <button type="button" onClick={toggleStar}>
                {watchlist ? (
                  <AiFillStar size={36} color="gold" />
                ) : (
                  <AiOutlineStar size={36} color="gold" />
                )}
              </button>
            </div>
            <div className="md:flex md:flex-col md:justify-center">
              <div className="flex mb-7 md:self-center">
                {stockData ? (
                  <img
                    className="h-10 w-10 md:h-20 md:w-20 self-center object-contain text-sm"
                    alt="company logo"
                    src={stockData.logo}
                  />
                ) : (
                  <div className="h-10 w-10 md:h-20 md:w-20 self-center object-contain flex justify-center">
                    <div className="self-center">
                      <PulseLoader
                        loading={loadingHistoricalData}
                        size={5}
                        aria-label="Loading Spinner"
                        color="#6638B3"
                      />
                    </div>
                  </div>
                )}
                <span className="ml-2 self-center">
                  <h2 className="text-lg font-bold md:text-3xl">
                    {stockData ? stockData.companyName : 'Loading...'}
                  </h2>
                  <p className="text-xs md:text-xl text-symbol">
                    {stockData ? stockData.ticker.toUpperCase() : 'Loading...'}
                  </p>
                </span>
              </div>
              <div className="md:self-center">
                <p className="text-neutraldark self-center px-8 md:px-2 md:mb-5 lg:ml-4 lg:text-lg md:hidden">
                  Price{' '}
                  <span className="text-white self-center float-right text-lg relative bottom-1 md:hidden">
                    {stockData
                      ? `$${stockData && stockData.price.toFixed(2)}`
                      : 'Loading...'}
                  </span>
                </p>
                <p
                  className="md:text-2xl lg:text-3xl md:text-white md:block"
                  hidden
                >
                  {stockData ? `$${stockData.price.toFixed(2)}` : 'Loading...'}
                </p>
              </div>
            </div>
            <ul className="grid grid-cols-2 w-full gap-4 mt-8 md:grid-cols-4">
              <li className="p-4 col-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                  Market Cap
                </h2>
                <p>
                  {stockData
                    ? `$${formatMarketCap(stockData && stockData.marketCap)}`
                    : 'Loading...'}
                </p>
              </li>
              <li className="p-4 col-span-1 md:row-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                  Previous Close
                </h2>
                <p>
                  {stockData
                    ? `$${stockData.previousClose.toFixed(2)}`
                    : 'Loading...'}
                </p>
              </li>
              <li className="p-4 col-span-1 md:row-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                  52W Range
                </h2>
                <p>
                  {stockData
                    ? `$${stockData.weekLow.toFixed(
                        2
                      )}-$${stockData.weekHigh.toFixed(2)}`
                    : 'Loading...'}
                </p>
              </li>
              <li className="p-4 col-span-1 md:row-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark text-sm mb-8 lg:text-lg">
                  Stock Prediction
                </h2>
                <div className="flex flex-row justify-between">
                  <p className="flex flex-row">
                    {stockPrediction ? (
                      <>
                        {(stockPrediction.confidence * 100).toFixed(2)}%{' '}
                        {stockPrediction.direction === 'up' ? (
                          <span className="text-green-900 self-center ml-2">
                            <AiOutlineArrowUp size={20} />
                          </span>
                        ) : (
                          <span className="text-red-900 self-center ml-2">
                            <AiOutlineArrowDown size={24} />
                          </span>
                        )}
                      </>
                    ) : (
                      'Loading...'
                    )}
                  </p>
                  <span className="self-center hover:text-primarydark">
                    <AiOutlineInfoCircle
                      size={20}
                      onClick={() => {
                        setPopUpHidden(!popUpHidden);
                      }}
                    />
                  </span>
                </div>
                {!popUpHidden && (
                  <div className="flex flex-col absolute text-white text-xs mt-5 text-center mr-20 border-primarydark border-2 bg-background p-4 rounded-xl select-none opacity-70">
                    <p className="text-xs">
                      Generated from a model trained to predict the stock trend
                      for today, by the close of the day. The prediction
                      encompasses the stock trend assigned by the model and its
                      degree of certainty in its prediction. Models leverage
                      valuable insights from sentiment analysis, technical
                      indicators and historical data to deliver predictions.
                    </p>
                  </div>
                )}
              </li>
            </ul>
          </div>
          <div className="mt-5 p-5 mx-[16px] font-display bg-main rounded-t-xl">
            <ul className="flex flex-row gap-4 justify-end mb-4">
              {Object.keys(chartConfig).map((item) => {
                return (
                  <li key={item}>
                    <ChartFilter
                      text={item}
                      active={filter === item}
                      disabled={closedMarket}
                      onClick={() => {
                        setFilter(item);
                      }}
                    />
                  </li>
                );
              })}
            </ul>
          </div>
          <div className="flex font-display text-white bg-main rounded-xl rounded-t-none mx-[16px] pl-0">
            {!loadingHistoricalData ? (
              <ResponsiveContainer width="99%" height={350}>
                <AreaChart
                  data={historicalData[filter]}
                  margin={{ top: 0, right: 0, left: 0, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6638B3" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#6638B3" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="date" hide />
                  <YAxis
                    dataKey="value"
                    domain={['dataMin', 'dataMax']}
                    allowDecimals={false}
                    tickLine={false}
                    scale="sequential"
                  />
                  <Tooltip
                    coordinate={{ y: 0 }}
                    content={<CustomTooltip />}
                    cursor={{ stroke: '#6638B3', strokeWidth: 1 }}
                  />
                  <Area
                    dataKey="value"
                    type="monotone"
                    stroke="#6638B3"
                    fill="url(#gradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[350px] w-full flex items-center justify-center">
                <PulseLoader
                  loading={!loadingHistoricalData}
                  size={10}
                  aria-label="Loading Spinner"
                  color="#6638B3"
                />
              </div>
            )}
          </div>
        </div>
        <div className="mt-5 bg-main rounded-xl lg:col-span-1 lg:mt-10 mx-[16px] lg:ml-0">
          <div className="flex flex-row font-display text-xl font-bold md:text-2xl justify-center mt-5 pt-5 lg:pt-0">
            <button
              type="button"
              className={`${
                activeTab === 'buy'
                  ? 'text-primary border-primary'
                  : 'text-neutraldark'
              } border-b-4 px-7 md:px-8 2xl:px-16 pb-4`}
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
              } border-b-4 px-7 md:px-8 2xl:px-16 pb-4`}
              onClick={() => setActiveTab('sell')}
            >
              Sell
            </button>
          </div>
          {stockData && (
            <BuySellForm
              type={activeTab}
              logo={stockData.logo}
              price={stockData.price}
              ticker={stockData.ticker}
              loading={loadingHistoricalData}
            />
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default IndividualCompany;
