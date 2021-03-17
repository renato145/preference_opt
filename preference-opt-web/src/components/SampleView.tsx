import React, { HTMLProps } from "react";

interface Props extends Omit<HTMLProps<HTMLDivElement>, "data"> {
  data: Float64Array;
}

export const SampleView: React.FC<Props> = ({ data, ...props }) => {
  const sample = Array.from(data).map((o) => (o*255).toFixed());
  const color = `rgb(${sample})`;

  return (
    <div {...props}>
      <div className="w-48 h-48 border-4 border-gray-700" style={{ backgroundColor: color }} />
    </div>
  );
};
