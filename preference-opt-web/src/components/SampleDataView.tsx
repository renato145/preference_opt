import React, { HTMLProps } from "react";
import { SampleSelector } from "./SampleSelector";
import { BestSampleView } from "./BestSampleView";
import { TStore, useStore } from "../store";

const selector = ({ sampleData, savedSamples, bestSample }: TStore) => ({
  sampleData,
  savedSamples,
  bestSample,
});

export const SampleDataView: React.FC<HTMLProps<HTMLDivElement>> = ({
  ...props
}) => {
  const { sampleData, savedSamples, bestSample } = useStore(selector);

  return (
    <div {...props}>
      {sampleData !== undefined ? (
        <div className="flex flex-col items-center">
          <div className="flex justify-around">
            <SampleSelector idx={sampleData.idx1} data={sampleData.sample1()} />
            <SampleSelector
              className="ml-8"
              idx={sampleData.idx2}
              data={sampleData.sample2()}
            />
          </div>
          <div className="mt-10 flex flex-wrap justify-center space-x-2">
            {savedSamples.map((data, i) => (
              <BestSampleView key={i} data={data} />
            ))}
            <BestSampleView data={bestSample} active />
          </div>
        </div>
      ) : null}
    </div>
  );
};
