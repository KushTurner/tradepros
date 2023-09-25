/* eslint-disable react/jsx-key */
import Footer from './Footer';

const leaderboard = [
  { rank: 1, username: 'User1', balance: '$101000', return: '10%' },
  { rank: 2, username: 'User2', balance: '$100500', return: '5%' },
  { rank: 3, username: 'User3', balance: '$100000', return: '0%' },
];

function TradeHistory() {
  return (
    <div className="font-display">
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
            Leaderboard
          </h1>
          <p className="text-neutral">Lorem Ipsum is dummy text of printing.</p>
        </div>
        <div>
          <table className="table-fixed w-full border-b-2 border-border text-sm md:text-base">
            <thead className="table-header-group border-b-2 border-t-2 border-border">
              <tr className="table-row text-neutraldark text-left">
                <th className="table-cell">Rank</th>
                <th className="table-cell">Username</th>
                <th className="table-cell">Balance</th>
                <th className="table-cell">Return</th>
              </tr>
            </thead>
            <tbody className="table-row-group text-white">
              {leaderboard.map((data) => (
                <tr className="table-row">
                  <td className="table-cell">
                    <p className="pl-3">{data.rank}</p>
                  </td>
                  <td className="table-cell">{data.username}</td>
                  <td className="table-cell">{data.balance}</td>
                  <td className="table-cell">
                    <p className="pl-3">{data.return}</p>
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
