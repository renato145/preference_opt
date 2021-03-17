import React, { HTMLProps } from "react";
import { TStore, useStore } from "../store";
import { SampleView } from "./SampleView";

interface Props extends Omit<HTMLProps<HTMLDivElement>, "data"> {
  idx: number;
  data: Float64Array;
}

const selector = ({ selectSample }: TStore) => selectSample;

export const SampleSelector: React.FC<Props> = ({ idx, data, ...props }) => {
  const selectSample = useStore(selector);

  return (
    <div {...props}>
      <div className="flex flex-col items-center">
        <p className="text-xl font-bold">Sample {idx}</p>
        <SampleView className="mt-1" data={data} />
        <button
          className="mt-2 px-4 btn"
          onClick={() => selectSample(idx)}
        >
          Select this sample
        </button>
      </div>
    </div>
  );
};
