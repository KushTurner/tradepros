/* eslint-disable react/jsx-key */
import { useMediaQuery } from 'react-responsive';
import { useNavigate } from 'react-router-dom';
import { IoIosRemoveCircleOutline } from 'react-icons/io';
import { SyntheticEvent, useState } from 'react';
import Footer from './Footer';

function Watchlist() {
  const [watchlist, setWatchlist] = useState([
    {
      id: 0,
      companyName: 'Facebook',
      ticker: 'META',
      price: '$7196',
      change: 1.61,
      high: '$6957',
      low: '$4722',
    },
    {
      id: 1,
      companyName: 'Amazon',
      ticker: 'AMZN',
      price: 33,
      change: 1.61,
      high: '$6957',
      low: '$4722',
    },
    {
      id: 2,
      companyName: 'Apple',
      ticker: 'AAPL',
      price: 33,
      change: 1.61,
      high: '$6957',
      low: '$4722',
    },
  ]);

  const toggleStar = (id: number) => {
    setWatchlist((prevWatchlist) =>
      prevWatchlist.filter((item) => item.id !== id)
    );
  };

  const isLaptopAndDesktop = useMediaQuery({ query: '(min-width: 768px)' });
  const navigate = useNavigate();
  return (
    <div>
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
            Watchlist
          </h1>
          <p className="text-neutral">Lorem Ipsum is dummy text of printing.</p>
        </div>
        <div>
          <table className="table w-full border-b-2 border-border text-sm md:text-base font-display">
            <thead className="table-header-group border-b-2 border-t-2 border-border">
              <tr className="table-row text-neutraldark text-left">
                <th className="table-cell">Company</th>
                <th className="table-cell">Stock</th>
                <th className="table-cell">Price</th>
                <th className="table-cell">24%</th>
                {isLaptopAndDesktop && <th className="table-cell">24h High</th>}
                {isLaptopAndDesktop && <th className="table-cell">24h Low</th>}
                <th className="w-[4%] text-main">#</th>
              </tr>
            </thead>
            <tbody className="table-row-group text-white">
              {watchlist.map((data) => (
                <tr
                  className="table-row hover:bg-black hover:bg-opacity-30 hover:cursor-pointer"
                  onClick={() => navigate(`/stock/${data.ticker}`)}
                  key={data.id}
                >
                  <td className="table-cell w-[15%]">{data.companyName}</td>
                  <td className="table-cell w-[15%]">{data.ticker}</td>
                  <td className="table-cell w-[15%]">{data.price}</td>
                  <td className="table-cell w-[15%]">{data.change}</td>
                  {isLaptopAndDesktop && (
                    <td className="table-cell w-[15%]">{data.high}</td>
                  )}
                  {isLaptopAndDesktop && (
                    <td className="table-cell w-[15%]">{data.low}</td>
                  )}
                  <td className="w-[4%]">
                    <IoIosRemoveCircleOutline
                      size={isLaptopAndDesktop ? 24 : 20}
                      className="text-warning hover:opacity-30"
                      onClick={(e: SyntheticEvent) => {
                        e.stopPropagation();
                        toggleStar(data.id);
                      }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default Watchlist;
