import { ChartFilterProps } from '../constants/config';

function ChartFilter({ text, active, disabled, onClick }: ChartFilterProps) {
  const noData = disabled && text === '1D'; // On some days (Saturday and Sunday), 1D chart wont be available so disable 1D filter

  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-md font-display text-xs md:text-sm border-[1px] p-2 ${
        active
          ? 'bg-primary text-white border-primary'
          : 'text-gray-500 border-gray-500'
      } ${noData && 'opacity-20'}`}
      disabled={noData}
    >
      {text}
    </button>
  );
}

export default ChartFilter;
