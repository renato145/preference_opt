import React, { HTMLProps } from "react";
import { SampleSelector } from "./SampleSelector";
import { BestSampleView } from "./BestSampleView";
import { TStore, useStore } from "../store";

const selector = ({ sampleData }: TStore) => sampleData;

export const SampleDataView: React.FC<HTMLProps<HTMLDivElement>> = ({
  ...props
}) => {
  const data = useStore(selector);

  return (
    <div {...props}>
      {data !== undefined ? (
        <div className="flex flex-col items-center">
          <div className="flex justify-around">
            <SampleSelector idx={data.idx1} data={data.sample1()} />
            <SampleSelector className="ml-8" idx={data.idx2} data={data.sample2()} />
          </div>
          <BestSampleView className="mt-10" />
        </div>
      ) : null}
    </div>
  );
};
