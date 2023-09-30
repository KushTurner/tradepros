/* eslint-disable react-hooks/exhaustive-deps */
import { PulseLoader } from 'react-spinners';
import { useContext, useEffect, useState } from 'react';
import axios from 'axios';
import { BuySellFormTypeParams, PortfolioObject } from '../constants/config';
import { AuthContext } from '../context/AuthContext';

function BuySellForm({
  type,
  logo,
  ticker,
  price,
  loading,
}: BuySellFormTypeParams) {
  const { currentUser } = useContext(AuthContext);
  const [profile, setProfile] = useState<PortfolioObject>();
  const [amount, setAmount] = useState('');
  const [investment, setInvestment] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

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
      axios.get(`/transaction?symbol=${ticker}`, config).then(({ data }) => {
        setInvestment(data);
      });
    });
  }, [currentUser]);

  const purchase = () => {
    currentUser?.getIdToken().then((token) => {
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };
      const payload = {
        price: parseFloat(amount),
        stock: ticker,
      };
      axios
        .post(`/stock/${type}`, payload, config)
        .then(() => {
          setSuccess('Success');
          setTimeout(() => {
            setSuccess('');
          }, 2000);
          axios.get(`/transaction?symbol=${ticker}`, config).then((resp) => {
            setInvestment(resp.data);
          });
        })
        .catch((err) => {
          setError(err.response.data.message);
          setTimeout(() => {
            setError('');
          }, 2000);
        });
    });
  };

  return (
    <div className="pb-10 lg:pb-0 md:mt-0 lg:mt-36">
      <div className="flex justify-center mt-10">
        {!loading ? (
          <img
            className="h-16 w-16 self-center object-contain text-white font-display text-sm"
            alt="company logo"
            src={logo}
          />
        ) : (
          <div className="flex justify-center h-16 w-16">
            <div className="self-center">
              <PulseLoader
                loading={loading}
                size={5}
                aria-label="Loading Spinner"
                color="#6638B3"
              />
            </div>
          </div>
        )}
        <span className="ml-2">
          <h2 className="text-2xl font-bold text-white">
            {type === 'buy' ? 'BUY' : 'SELL'} {ticker}
          </h2>
          <p className="text-3xl font-bold text-symbol">${price.toFixed(2)}</p>
        </span>
      </div>
      <div className="flex flex-row justify-around text-white mt-10 font-display text-xl font-bold">
        <p>{type === 'buy' ? 'Balance' : 'Own'}</p>
        {type === 'buy' ? (
          <p>{profile && `$${profile.balance.toFixed()}`}</p>
        ) : (
          <p>{investment ? `$${investment}` : '$0'}</p>
        )}
      </div>
      <form className="flex flex-col mt-10 px-5 md:px-20 lg:px-5 gap-5">
        <h2 className="font-bold text-white text-center">Amount ($)</h2>
        <input
          type="number"
          placeholder="$"
          className="bg-background rounded-md text-white caret-primary focus:outline-none focus:border focus:border-primary pl-2 py-3 md:py-4 md:pl-3"
          onChange={(e) => {
            setAmount(e.target.value);
          }}
        />
        {error ? (
          <div className="text-sm md:text-xl font-display text-warning justify-center items-center">
            <p className="text-center self-center">{error}</p>
          </div>
        ) : null}
        {success ? (
          <div className="text-sm md:text-xl font-display text-success justify-center items-center">
            <p className="text-center self-center">{success}</p>
          </div>
        ) : null}
        <button
          type="button"
          onClick={purchase}
          className="rounded-2xl bg-primary font-display font-bold text-white p-3 text-lg md:text-2xl mt-8 md:mt-12 lg:mt-20"
        >
          {type === 'buy' ? 'BUY' : 'SELL'}
        </button>
      </form>
    </div>
  );
}

export default BuySellForm;
