/* eslint-disable react/no-array-index-key */
/* eslint-disable react-hooks/exhaustive-deps */
import axios from 'axios';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AiOutlineSearch } from 'react-icons/ai';
import useDebounce from '../hooks/useDebounce';
import { SearchObject } from '../constants/config';
import Footer from './Footer';

function Discover() {
  const navigate = useNavigate();
  const [search, setSearch] = useState<string>('');
  const debouncedSearch = useDebounce(search, 700);
  const [response, setResponse] = useState<SearchObject[]>();

  useEffect(() => {
    if (debouncedSearch.trim() === '') {
      setResponse([]);
      return;
    }
    axios.get(`search?keywords=${debouncedSearch}`).then(({ data }) => {
      setResponse(data.bestMatches);
    });
  }, [debouncedSearch]);

  return (
    <div className="font-display">
      <div className="bg-main mt-5 mx-[16px] md:mx-6 lg:mx-8 rounded-xl">
        <div className="flex flex-col">
          <div className="font-display mb-6 md:ml-10 lg:ml-16 mt-5 md:mt-10">
            <h1 className="text-white mb-1 text-2xl font-bold md:mb-3">
              Discover
            </h1>
            <p className="text-neutral">
              Search for companies you are looking to invest into
            </p>
          </div>
          <div className="">
            <div className="flex flex-row justify-center">
              <AiOutlineSearch
                size={24}
                className="relative left-9 self-center"
                color="#6638B3"
              />
              <input
                type="text"
                placeholder="Search company, eg. AAPL"
                className="bg-background rounded-md text-white caret-primarydark focus:outline-none focus:border focus:border-primarydark pl-12 py-2 md:py-3 w-4/5 md:pl-12"
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>
          <div className="w-4/5 flex justify-center self-center mb-16 mt-4">
            <div className="z-10 absolute w-3/4 shadow-inner shadow-gray-700">
              <ul className="flex flex-col">
                {response &&
                  response.slice(0, 7).map((data, index) => (
                    <li className="font-display bg-background" key={index}>
                      <button
                        type="button"
                        className={`flex flex-col w-full ${
                          index !== 0 ? 'border-t border-border' : ''
                        }`}
                        onClick={() => {
                          navigate(`/stock/${data['1. symbol']}`);
                        }}
                      >
                        <h2 className="font-bold text-primarydark pl-4 mt-2 text-xl">
                          {data['1. symbol']}
                        </h2>
                        <p className="text-primary text-sm pl-4 mb-2">
                          {data['2. name']}
                        </p>
                      </button>
                    </li>
                  ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default Discover;
