/* eslint-disable react/jsx-key */
import Footer from './Footer';

const historyData = [
  {
    name: 'META',
    action: 'Buy',
    amount: '$1000',
    quantity: '2',
    date: '29/08/23',
  },
  {
    name: 'AAPL',
    action: 'Sell',
    amount: '$1000',
    quantity: '3',
    date: '28/08/23',
  },
  {
    name: 'AMZN',
    action: 'Buy',
    amount: '$1000',
    quantity: '4',
    date: '27/08/23',
  },
];

function TradeHistory() {
  return (
    <div className="font-display">
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-10 lg:ml-16 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
            Trade History
          </h1>
          <p className="text-neutral">Lorem Ipsum is dummy text of printing.</p>
        </div>
        <div>
          <table className="table-fixed w-full border-b-2 border-border text-sm md:text-base">
            <thead className="table-header-group border-b-2 border-t-2 border-border">
              <tr className="table-row text-neutraldark text-left">
                <th className="table-cell">Stock</th>
                <th className="table-cell">Action</th>
                <th className="table-cell">Amount</th>
                <th className="table-cell">Quantity</th>
                <th className="table-cell">Date</th>
              </tr>
            </thead>
            <tbody className="table-row-group text-white">
              {historyData.map((data) => (
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
                  <td className="table-cell">
                    <p className="pl-5">{data.quantity}</p>
                  </td>
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
