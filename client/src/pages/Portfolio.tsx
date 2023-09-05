import { TfiWallet } from 'react-icons/tfi';
import PortfolioTable from '../components/PortfolioTable';
import { portfolioColumns } from '../components/columns';
import data from '../MOCK_DATA_2.json';
import Footer from './Footer';

function Portfolio() {
  return (
    <div>
      <div className="font-display text-white bg-main rounded-xl mt-5 mx-[16px] p-5 md:pb-10 md:mt-10">
        <div className="mb-10 md:mb-5 md:ml-2 lg:ml-6">
          <h1 className="text-2xl font-bold mb-2">Portfolio</h1>
          <p className="text-neutral text-sm">Update 16/02/2022 at 02:30PM</p>
        </div>
        <div className="md:grid md:grid-cols-10 lg:grid-cols-5">
          <div className="md:col-span-3 lg:md:col-span-1 md:flex md:justify-center md:self-center">
            <div className="flex flex-row justify-around md:flex-col">
              <div className="flex flex-row mb-10 md:mb-0">
                <TfiWallet size={16} className="self-center md:hidden" />
                <TfiWallet size={20} className="self-center hidden md:block" />
                <p className="text-neutraldark ml-3 self-center text-sm md:px-0 md:text-lg">
                  Available Balance
                </p>
              </div>
              <p className="text-white font-bold font-display text-xl text-center md:mt-3 md:text-2xl">
                $32,455.12
              </p>
            </div>
          </div>
          <div className="md:col-span-7 lg:col-span-4">
            <ul className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <li className="p-4 col-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark md:text-sm mb-8 lg:text-lg">
                  Total Investment
                </h2>
                <p>$30,455.12</p>
              </li>
              <li className="p-4 col-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark md:text-sm mb-8 lg:text-lg">
                  Total Return
                </h2>
                <p>$30,455.12</p>
              </li>
              <li className="p-4 col-span-2 md:col-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark md:text-sm mb-8 lg:text-lg">
                  Profit / Loss
                </h2>
                <p>$22,822,762,169</p>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
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
