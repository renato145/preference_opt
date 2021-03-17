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
      <div className="flex space-x-4">
        <p className="font-bold">Sample {idx}: </p>
        <SampleView data={data} />
        <button
          className="btn-confirm btn-sm"
          onClick={() => selectSample(idx)}
        >
          Select this sample
        </button>
      </div>
    </div>
  );
};
