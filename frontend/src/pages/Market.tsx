import CompanyTable from '../components/CompanyTable';
import { companyColumns } from '../components/columns';
import data from '../MOCK_DATA.json';
import Footer from './Footer';

function Market() {
  return (
    <div>
      <div className="bg-[#121318] rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold">Companies</h1>
          <p className="text-[#808591]">
            Lorem Ipsum is dummy text of printing.
          </p>
        </div>
        <CompanyTable data={data} columns={companyColumns} />
      </div>
      <Footer />
    </div>
  );
}

export default Market;
