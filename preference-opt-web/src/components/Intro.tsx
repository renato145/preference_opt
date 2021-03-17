import React, { HTMLProps, useCallback } from "react";
import { TState, TStore, useStore } from "../store";

const selector = ({ loadSampleData, state, setState }: TStore) => ({
  loadSampleData,
  state,
  setState,
});

export const Intro: React.FC<HTMLProps<HTMLDivElement>> = ({ ...props }) => {
  const { loadSampleData, state, setState } = useStore(selector);
  const startEvent = useCallback(() => {
    loadSampleData();
    setState(TState.Running);
  }, [loadSampleData, setState]);

  return (
    <div {...props}>
      {state !== TState.Running ? (
        <div className="mx-auto max-w-2xl flex flex-col text-lg break-words">
          <p>
            This is an example of active preference optimization using the
            gaussian processes{" "}
            <a
              href="http://mlg.eng.cam.ac.uk/zoubin/papers/icml05chuwei-pl.pdf"
              target="_black"
              rel="noopener"
            >
              "Preference Learning with Gaussian Processes"
            </a>
            . I adapted the python implementation{" "}
            <a href="https://github.com/chariff/GPro">GPro</a> to Rust so it's
            possible to load it in here using{" "}
            <a href="https://webassembly.org/" target="_black" rel="noopener">
              Wasm
            </a>
            .
          </p>
          <p className="mt-2">
            This is a simple example where you select a group of colours. On
            each iteration 2 colour samples are shown and you can select which
            one your prefer. At any point you can choose to keep the current
            color.
          </p>
          <p className="mt-4">
            You can find the code here:{" "}
            <a
              href="https://github.com/renato145/preference_opt"
              target="_black"
              rel="noopener"
            >
              https://github.com/renato145/preference_opt
            </a>
            .
          </p>
          {state === TState.Initial ? (
            <button
              className="mt-6 md:mt-12 md:self-center md:px-40 btn-confirm btn-lg"
              onClick={startEvent}
            >
              Start
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};
