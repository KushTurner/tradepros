/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable react/no-array-index-key */
import { useMediaQuery } from 'react-responsive';
import { useNavigate } from 'react-router-dom';
import { IoIosRemoveCircleOutline } from 'react-icons/io';
import { SyntheticEvent, useContext, useEffect, useState } from 'react';
import axios from 'axios';
import Footer from './Footer';
import { AuthContext } from '../context/AuthContext';
import { WatchlistObject } from '../constants/config';

function Watchlist() {
  const [loadingWatchlist, setLoadingWatchlist] = useState(false);
  const [watchlist, setWatchlist] = useState<WatchlistObject[]>([]);
  const isLaptopAndDesktop = useMediaQuery({ query: '(min-width: 768px)' });
  const navigate = useNavigate();

  const { currentUser } = useContext(AuthContext);

  useEffect(() => {
    setLoadingWatchlist(true);
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      axios.get('/watchlist', config).then(({ data }) => {
        setWatchlist(data);
      });
      setLoadingWatchlist(false);
    });
  }, [currentUser]);

  const toggleStar = (ticker: string) => {
    setWatchlist((prevWatchlist) =>
      prevWatchlist.filter((item) => item.ticker !== ticker)
    );
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      axios.delete(`/watchlist/${ticker}`, config);
    });
  };

  return (
    <div>
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
            Watchlist
          </h1>
          <p className="text-neutral">
            Comapanies you are currently interested in
          </p>
        </div>
        <div>
          <table className="table w-full border-b-2 border-border text-sm md:text-base font-display">
            <thead className="table-header-group border-b-2 border-t-2 border-border">
              <tr className="table-row text-neutraldark text-left">
                <th className="table-cell">Logo</th>
                <th className="table-cell">Name</th>
                <th className="table-cell">Ticker</th>
                <th className="table-cell">Industry</th>
                <th className="w-[4%] text-main">#</th>
              </tr>
            </thead>
            {!loadingWatchlist && (
              <tbody className="table-row-group text-white">
                {watchlist &&
                  watchlist.map((data, index) => (
                    <tr
                      className="table-row hover:bg-black hover:bg-opacity-30 hover:cursor-pointer"
                      onClick={() => navigate(`/stock/${data.ticker}`)}
                      key={index}
                    >
                      <td className="table-cell w-[15%]">
                        <img
                          alt="Some logo init"
                          src={data.logo}
                          className="self-center object-contain w-10 h-10"
                        />
                      </td>
                      <td className="table-cell w-[15%]">{data.name}</td>
                      <td className="table-cell w-[15%]">{data.ticker}</td>
                      <td className="table-cell w-[15%]">
                        {data.finnhubIndustry}
                      </td>
                      <td className="w-[4%]">
                        <IoIosRemoveCircleOutline
                          size={isLaptopAndDesktop ? 24 : 20}
                          className="text-warning hover:opacity-30"
                          onClick={(e: SyntheticEvent) => {
                            e.stopPropagation();
                            toggleStar(data.ticker);
                          }}
                        />
                      </td>
                    </tr>
                  ))}
              </tbody>
            )}
          </table>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default Watchlist;
