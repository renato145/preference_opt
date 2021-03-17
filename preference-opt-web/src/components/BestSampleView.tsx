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
      <div className="flex flex-col items-center">
        <p className="text-xl font-bold">Best sample</p>
        {bestSample !== undefined ? (
          <>
            <SampleView className="mt-1" data={bestSample} />
            <button className="mt-2 px-4 btn">Select this sample</button>
          </>
        ) : null}
      </div>
    </div>
  );
};
