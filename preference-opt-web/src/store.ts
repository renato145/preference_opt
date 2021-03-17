import create from "zustand";
import { OptimizationEngine, SampleData } from "preference-opt-wasm";

export enum TState {
  Initial,
  Running,
  Help,
}

export type TStore = {
  step: number;
  engine: OptimizationEngine;
  state: TState;
  sampleData?: SampleData;
  bestSample?: Float64Array;
  savedSamples: Float64Array[];
  loadSampleData: () => void;
  selectSample: (idx: number) => void;
  saveCurrentBest: () => void;
  setState: (state: TState) => void;
  toogleHelp: () => void;
};

const newEngine = () =>
  OptimizationEngine.new_empty(3).with_same_bounds(new Float64Array([0, 1]));

export const useStore = create<TStore>((set, get) => ({
  step: 0,
  engine: newEngine(),
  state: TState.Initial,
  savedSamples: [],
  loadSampleData: () => {
    set(({ step, engine, sampleData }) => {
      // TODO: There is an error when doing many steps
      const prior = step < 12 ? sampleData?.f_prior() : undefined;
      return {
        step: step + 1,
        sampleData: engine.get_next_sample(50, 1, prior),
      };
    });
  },
  selectSample: (idx) => {
    const other =
      get().sampleData?.idx1 === idx
        ? get().sampleData?.idx2
        : get().sampleData?.idx1;

    if (other === undefined) return;

    get().engine.add_preference(idx, other);
    const bestSample = get().engine.get_optimal_values();
    set({ bestSample });
    get().loadSampleData();
  },
  saveCurrentBest: () => {
    let bestSample = get().bestSample;
    if (bestSample === undefined) return;
    get().savedSamples.push(bestSample);
    set({
      step: 0,
      engine: newEngine(),
      bestSample: undefined,
    });
    get().loadSampleData();
  },
  setState: (state) => set({ state }),
  toogleHelp: () =>
    set(({ state }) => ({
      state: state === TState.Help ? TState.Running : TState.Help,
    })),
}));
