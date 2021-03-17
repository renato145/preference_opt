import React, { HTMLProps } from "react";

interface Props extends Omit<HTMLProps<HTMLDivElement>, "data"> {
  data: Float64Array;
}

export const SampleView: React.FC<Props> = ({ data, ...props }) => {
  const sample = Array.from(data).map((o) => o.toFixed(2));

  return (
    <div {...props}>
      <p>[{sample.join(", ")}]</p>
    </div>
  );
};
