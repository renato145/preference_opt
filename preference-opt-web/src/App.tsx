import React from "react";
import { Intro } from "./components/Intro";
import { SampleDataView } from "./components/SampleDataView";
import { TState, TStore, useStore } from "./store";

const selector = ({ state, toogleHelp }: TStore) => ({ state, toogleHelp });

export const App = () => {
  const { state, toogleHelp } = useStore(selector);

  return (
    <div className="container mx-auto px-2 md:px-20 py-5">
      <p className="text-3xl font-extrabold text-center">
        Preference Optimization{" "}
        {state !== TState.Initial ? (
          <button className="btn-link text-lg" onClick={toogleHelp}>
            (info)
          </button>
        ) : null}
      </p>
      <Intro className={ state === TState.Initial ? "mt-6 md:mt-12" : "mt-4"} />
      <SampleDataView className="mt-10" />
    </div>
  );
};
