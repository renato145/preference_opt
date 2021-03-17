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
        <>
          <SampleSelector idx={data.idx1} data={data.sample1()} />
          <SampleSelector
            className="mt-2"
            idx={data.idx2}
            data={data.sample2()}
          />
          <BestSampleView className="mt-4" />
        </>
      ) : null}
    </div>
  );
};
