import React, { HTMLProps } from "react";
import { TStore, useStore } from "../store";
import { array2color } from "../utils";
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
  const title = data !== undefined ? array2color(data) : "rgb(...)";

  return (
    <div {...props}>
      <div className="flex flex-col items-center">
        <p className="text-xl font-bold">{title}</p>
        <SampleView className="mt-1" data={data} />
        {active && data !== undefined ? (
          <button className="mt-2 px-4 btn" onClick={saveCurrentBest}>
            Select this sample
          </button>
        ) : null}
      </div>
    </div>
  );
};
