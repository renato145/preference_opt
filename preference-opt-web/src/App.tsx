import React from "react";
import { Intro } from "./components/Intro";
import { SampleDataView } from "./components/SampleDataView";
import { OptimizationState, TStore, useStore } from "./store";

const selector = ({ optState }: TStore) => optState;

export const App = () => {
  const optState = useStore(selector);

  return (
    <div className="container mx-auto px-20 py-5">
      <p className="text-3xl font-extrabold text-center">Preference Optimization</p>
      <div className="mt-10">
        {optState === OptimizationState.Initial ? (
          <Intro />
        ) : (
          <SampleDataView className="" />
        )}
      </div>
    </div>
  );
};
