import React, { HTMLProps } from "react";
import { array2color } from "../utils";

interface Props extends Omit<HTMLProps<HTMLDivElement>, "data"> {
  data?: Float64Array;
}

export const SampleView: React.FC<Props> = ({ data, ...props }) => {
  const color = data !== undefined ? array2color(data) : undefined;
  const style = color !== undefined ? { backgroundColor: color } : {};
  const title = color !== undefined ? color : "rgb(...)";

  return (
    <div {...props}>
      <p className="text-xl font-bold text-center">{title}</p>
      <div className="mt-1 w-48 h-48 border-4 border-gray-700" style={style} />
    </div>
  );
};
