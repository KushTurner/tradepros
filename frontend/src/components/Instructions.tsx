type InstructionsProps = {
  index: number;
  header: string;
  description: string;
};

function Instructions({ index, header, description }: InstructionsProps) {
  return (
    <li className="rounded-lg bg-[#080808] m-2 p-5 mb-4 pl-8">
      <h5 className="font-bold text-md mb-4">
        <span className="text-[#5367FE]">{`${index}.`}</span> {header}
      </h5>
      <p className="text-[#9395A5] text-sm">{description}</p>
    </li>
  );
}

export default Instructions;
