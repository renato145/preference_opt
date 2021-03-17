import React, { HTMLProps } from "react";
import { TStore, useStore } from "../store";
import { SampleView } from "./SampleView";

const selector = ({ bestSample }: TStore) => bestSample;

export const BestSampleView: React.FC<HTMLProps<HTMLDivElement>> = ({
  ...props
}) => {
  const bestSample = useStore(selector);

  return (
    <div {...props}>
      <div className="flex space-x-4">
        <p className="font-bold">Best sample: </p>
        {bestSample !== undefined ? <SampleView data={bestSample} /> : null}
      </div>
    </div>
  );
};
