/* eslint-disable react-hooks/exhaustive-deps */
/* eslint-disable react/jsx-key */
import { TfiWallet } from 'react-icons/tfi';
import { useNavigate } from 'react-router-dom';
import { useMediaQuery } from 'react-responsive';
import { useContext, useEffect, useState } from 'react';
import axios from 'axios';
import Footer from './Footer';
import { AuthContext } from '../context/AuthContext';
import { formatDate } from '../helpers/date-helper';
import { InvestmentObject, PortfolioObject } from '../constants/config';

function Portfolio() {
  const { currentUser } = useContext(AuthContext);
  const navigate = useNavigate();
  const isLaptopAndDesktop = useMediaQuery({ query: '(min-width: 768px)' });
  const [profile, setProfile] = useState<PortfolioObject>();
  const [totalInvestments, setTotalInvestments] = useState(0.0);
  const [investments, setInvestments] = useState<InvestmentObject[]>([]);
  const [loadingInvestments, setLoadingInvestments] = useState(false);

  useEffect(() => {
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      axios.get('/me', config).then(({ data }) => {
        setProfile(data);
      });
      setLoadingInvestments(true);
      axios.get('/transactions', config).then(({ data }) => {
        let counter = 0;
        if (data) {
          data.forEach((obj: InvestmentObject) => {
            counter += obj.investment;
          });
        }
        setInvestments(data);
        setTotalInvestments(counter);
        setLoadingInvestments(false);
      });
    });
  }, [currentUser]);

  return (
    <div>
      <div className="font-display text-white bg-main rounded-xl mt-5 mx-[16px] p-5 md:pb-10 md:mt-10">
        <div className="mb-10 md:mb-5 md:ml-2 lg:ml-6">
          <h1 className="text-2xl font-bold mb-2">
            {profile && `${profile.username}'s`} Portfolio
          </h1>
          <p className="text-neutral text-sm">
            {profile
              ? `Account Created ${formatDate(profile.created_at)}`
              : 'Loading...'}
          </p>
        </div>
        <div className="md:grid md:grid-cols-10 lg:grid-cols-5">
          <div className="md:col-span-3 lg:md:col-span-1 md:flex md:justify-center md:self-center">
            <div className="flex flex-row justify-around md:flex-col">
              <div className="flex flex-row mb-10 md:mb-0">
                <TfiWallet size={16} className="self-center md:hidden" />
                <TfiWallet size={20} className="self-center hidden md:block" />
                <p className="text-neutraldark ml-3 self-center md:px-0 md:text-lg">
                  Available Balance
                </p>
              </div>
              <p className="text-white font-bold font-display text-xl text-center md:mt-3 md:text-2xl">
                {profile ? `$${profile.balance.toFixed(2)}` : 'Loading...'}
              </p>
            </div>
          </div>
          <div className="md:col-span-7 lg:col-span-4">
            <ul className="grid grid-cols-2  gap-4">
              <li className="p-4 col-span-2 md:col-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark md:text-sm mb-8 lg:text-lg">
                  Total Return ($)
                </h2>
                <p>
                  {profile ? profile.balance_gain.toFixed(2) : 'Loading...'}
                </p>
              </li>
              <li className="p-4 col-span-2 md:col-span-1 bg-background rounded-md">
                <h2 className="text-neutraldark md:text-sm mb-8 lg:text-lg">
                  Total Investment ($)
                </h2>
                <p>{profile ? `${totalInvestments}` : 'Loading...'}</p>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div className="bg-main rounded-xl mt-5 mx-[16px] p-4 md:mt-10">
        <div className="font-display mb-6 md:ml-6 md:mt-3">
          <h1 className="text-white mb-1 text-2xl font-bold">
            Current Investments
          </h1>
        </div>
        <div>
          <table className="table-fixed w-full border-b-2 border-border text-sm md:text-base font-display">
            <thead className="table-header-group border-b-2 border-t-2 border-border">
              <tr className="table-row text-neutraldark text-left">
                <th className="table-cell">Stock</th>
                {isLaptopAndDesktop && <th className="table-cell">Units</th>}
                <th className="table-cell">Total Investment</th>
              </tr>
            </thead>
            <tbody className="table-row-group text-white">
              {!loadingInvestments &&
                investments &&
                investments.map((data) => (
                  <tr
                    className="table-row hover:bg-black hover:bg-opacity-30 hover:cursor-pointer"
                    onClick={() => navigate(`/stock/${data.name}`)}
                  >
                    <td className="table-cell">{data.name}</td>
                    {isLaptopAndDesktop && (
                      <td className="table-cell">
                        <p>{data.shares.toFixed(2)}</p>
                      </td>
                    )}
                    <td className="table-cell">${data.investment}</td>
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

export default Portfolio;
