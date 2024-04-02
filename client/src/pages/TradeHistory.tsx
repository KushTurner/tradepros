/* eslint-disable react/jsx-key */
import { useContext, useEffect, useState } from 'react';
import axios from 'axios';
import Footer from './Footer';
import { AuthContext } from '../context/AuthContext';
import { TradeHistoryObject } from '../constants/config';

function TradeHistory() {
  const { currentUser } = useContext(AuthContext);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [history, setHistory] = useState<TradeHistoryObject[]>();

  useEffect(() => {
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      axios.get('/history', config).then(({ data }) => {
        setHistory(data);
      });
    });
    setLoadingHistory(false);
  }, [currentUser]);

  return (
    <div className="font-display">
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
            Trade History
          </h1>
          <p className="text-neutral">The stocks you have previously sold</p>
        </div>
        <div>
          <table className="table-fixed w-full border-b-2 border-border text-sm md:text-base">
            <thead className="table-header-group border-b-2 border-t-2 border-border">
              <tr className="table-row text-neutraldark text-left">
                <th className="table-cell">Stock</th>
                <th className="table-cell">Action</th>
                <th className="table-cell">Amount</th>
                <th className="table-cell">Units</th>
                <th className="table-cell">Date</th>
              </tr>
            </thead>
            <tbody className="table-row-group text-white">
              {!loadingHistory &&
                history &&
                history.map((data) => (
                  <tr className="table-row">
                    <td className="table-cell">{data.name}</td>
                    <td
                      className={`table-cell ${
                        data.action === 'Buy' ? 'text-success' : 'text-warning'
                      }`}
                    >
                      {data.action}
                    </td>
                    <td className="table-cell">{data.amount}</td>
                    <td className="table-cell">{data.quantity.toFixed(2)}</td>
                    <td className="table-cell">{data.date}</td>
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
