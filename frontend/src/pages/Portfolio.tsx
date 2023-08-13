import PortfolioTable from '../components/PortfolioTable';
import { portfolioColumns } from '../components/columns';
import data from '../MOCK_DATA_2.json';
import Footer from './Footer';

function Portfolio() {
  return (
    <div>
      <div className="font-display text-white bg-[#121318] rounded-xl mt-5 mx-[16px] p-5 md:pb-20 md:mt-10">
        <div className="mb-10 md:ml-2 lg:ml-6">
          <h1 className="text-2xl font-bold mb-2">Portfolio</h1>
          <p className="text-[#9395A5]">Update 16/02/2022 at 02:30PM</p>
        </div>
        <p className="text-[#616573] px-8 text-sm md:px-2 md:mb-5 lg:ml-4 lg:text-lg">
          Available Balance{' '}
          <span className="text-white float-right text-lg relative bottom-1 md:hidden">
            $32,455.12
          </span>
        </p>
        <p
          className="md:text-2xl lg:text-3xl md:text-white md:block md:px-2 lg:ml-4"
          hidden
        >
          $32,455.12
        </p>
        <ul className="grid grid-cols-2 md:grid-cols-3 w-full gap-4 mt-8 md:w-9/12 md:float-right md:relative md:bottom-28">
          <li className="p-4 col-span-1 bg-[#080808] rounded-md">
            <h2 className="text-[#616573] md:text-sm mb-8 lg:text-lg">
              Total Investment
            </h2>
            <p>$30,455.12</p>
          </li>
          <li className="p-4 col-span-1 bg-[#080808] rounded-md">
            <h2 className="text-[#616573] md:text-sm mb-8 lg:text-lg">
              Total Return
            </h2>
            <p>$30,455.12</p>
          </li>
          <li className="p-4 col-span-2 md:col-span-1 bg-[#080808] rounded-md">
            <h2 className="text-[#616573] md:text-sm mb-8 lg:text-lg">
              Profit / Loss
            </h2>
            <p>$22,822,762,169</p>
          </li>
        </ul>
      </div>
      <div className="bg-[#121318] rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-6 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold">Investments</h1>
        </div>
        <PortfolioTable data={data} columns={portfolioColumns} />
      </div>

      <Footer />
    </div>
  );
}

export default Portfolio;
