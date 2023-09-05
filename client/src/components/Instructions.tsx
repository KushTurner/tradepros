type InstructionsProps = {
  index: number;
  header: string;
  description: string;
};

function Instructions({ index, header, description }: InstructionsProps) {
  return (
    <li className="rounded-lg bg-background m-2 p-5 mb-4 pl-8">
      <h5 className="font-bold text-md mb-4">
        <span className="text-primarydark">{`${index}.`}</span> {header}
      </h5>
      <p className="text-neutraldark text-sm">{description}</p>
    </li>
  );
}

export default Instructions;
