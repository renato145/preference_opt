import React, { HTMLProps } from "react";
import { TStore, useStore } from "../store";
import { SampleView } from "./SampleView";

interface Props extends Omit<HTMLProps<HTMLDivElement>, "data"> {
  data?: Float64Array;
  active?: boolean;
}

const selector = ({ saveCurrentBest }: TStore) => saveCurrentBest;

export const BestSampleView: React.FC<Props> = ({
  data,
  active = false,
  ...props
}) => {
  const saveCurrentBest = useStore(selector);

  return (
    <div {...props}>
      <div className="flex flex-col items-center">
        <SampleView className="mt-1" data={data} />
        {active && data !== undefined ? (
          <button className="mt-2 px-4 btn" onClick={saveCurrentBest}>
            Keep this!
          </button>
        ) : null}
      </div>
    </div>
  );
};
