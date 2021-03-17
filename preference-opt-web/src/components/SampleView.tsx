import React, { HTMLProps } from "react";
import { array2color } from "../utils";

interface Props extends Omit<HTMLProps<HTMLDivElement>, "data"> {
  data?: Float64Array;
}

export const SampleView: React.FC<Props> = ({ data, ...props }) => {
  const style =
    data !== undefined ? { backgroundColor: array2color(data) } : {};

  return (
    <div {...props}>
      <div className="w-48 h-48 border-4 border-gray-700" style={style} />
    </div>
  );
};
