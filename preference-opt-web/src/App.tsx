import React from "react";
import { Intro } from "./components/Intro";
import { SampleDataView } from "./components/SampleDataView";
import { OptimizationState, TStore, useStore } from "./store";

const selector = ({ optState }: TStore) => optState;

export const App = () => {
  const optState = useStore(selector);

  return (
    <div className="p-20 text-xl">
      {optState === OptimizationState.Initial ? (
        <Intro />
      ) : (
        <div className="mt-4 p-4 max-w-max bg-blue-100">
          <SampleDataView />
        </div>
      )}
    </div>
  );
};
