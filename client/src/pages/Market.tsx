function Market() {
  return (
    <div>
      <div className="bg-main mt-5 mx-[16px] md:mx-6 lg:mx-8 rounded-xl">
        <div className="flex justify-center">
          <input
            type="text"
            placeholder="Company, eg. AAPL"
            className="bg-background rounded-md text-white caret-primarydark focus:outline-none focus:border focus:border-primarydark pl-2 py-2 md:py-3 md:pl-3 w-4/5"
          />
        </div>
      </div>
    </div>
  );
}

export default Market;
