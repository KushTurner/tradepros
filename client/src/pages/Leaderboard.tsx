import LeaderboardTable from '../components/LeaderboardTable';
import { leadererboardColumns } from '../components/columns';
import data from '../MOCK_DATA_5.json';
import Footer from './Footer';

function TradeHistory() {
  return (
    <div>
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
            Trade History
          </h1>
          <p className="text-neutral">Lorem Ipsum is dummy text of printing.</p>
        </div>
        <LeaderboardTable data={data} columns={leadererboardColumns} />
      </div>
      <Footer />
    </div>
  );
}

export default TradeHistory;
