interface ChartFilterProps {
  text: string;
  active: boolean;
  onClick: () => void; // Assuming onClick is a function with no arguments
}

function ChartFilter({ text, active, onClick }: ChartFilterProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-md font-display text-xs md:text-sm border-[1px] p-2 ${
        active
          ? 'bg-primary text-white border-primary'
          : 'text-gray-500 border-gray-500'
      }`}
    >
      {text}
    </button>
  );
}

export default ChartFilter;
