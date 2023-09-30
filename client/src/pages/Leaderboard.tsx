/* eslint-disable react/jsx-key */
import { useContext, useEffect, useState } from 'react';
import axios from 'axios';
import Footer from './Footer';
import { AuthContext } from '../context/AuthContext';
import { LeaderboardObject } from '../constants/config';

function TradeHistory() {
  const { currentUser } = useContext(AuthContext);
  const [leaderboard, setLeaderboard] = useState<LeaderboardObject[]>([]);
  const [loadingLeaderboard, setLoadingLeaderboard] = useState(false);

  useEffect(() => {
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      setLoadingLeaderboard(true);
      axios.get('/leaderboard', config).then(({ data }) => {
        setLeaderboard(data);
        setLoadingLeaderboard(false);
      });
    });
  }, [currentUser]);

  return (
    <div className="font-display">
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
            Leaderboard
          </h1>
          <p className="text-neutral">
            Ranks users based on the highest return
          </p>
        </div>
        <div>
          <table className="table-fixed w-full border-b-2 border-border text-sm md:text-base">
            <thead className="table-header-group border-b-2 border-t-2 border-border">
              <tr className="table-row text-neutraldark text-left">
                <th className="table-cell">Rank</th>
                <th className="table-cell">Username</th>
                <th className="table-cell">Return</th>
              </tr>
            </thead>
            <tbody className="table-row-group text-white">
              {!loadingLeaderboard &&
                leaderboard &&
                leaderboard.map((data) => (
                  <tr className="table-row">
                    <td className="table-cell">
                      <p className="pl-3">{data.rank}</p>
                    </td>
                    <td className="table-cell">{data.username}</td>
                    <td className="table-cell">
                      <p>{((data.balance / 100000) * 100).toFixed(2)}%</p>
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

export default TradeHistory;
