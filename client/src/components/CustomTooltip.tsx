import { TooltipProps } from 'recharts';
import {
  ValueType,
  NameType,
} from 'recharts/types/component/DefaultTooltipContent';

function CustomTooltip({
  active,
  payload,
  label,
}: TooltipProps<ValueType, NameType>) {
  if (active) {
    return (
      <div>
        <ul className="flex flex-col bg-black bg-opacity-40 font-display p-2 md:p-4 rounded-lg">
          <li className="font-bold text-center text-lg text-primarydark">{`${payload?.[0].value}`}</li>
          <li className="text-xs text-neutraldark">{`${label}`}</li>
        </ul>
      </div>
    );
  }

  return null;
}

export default CustomTooltip;
