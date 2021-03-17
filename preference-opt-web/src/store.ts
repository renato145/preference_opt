import create from "zustand";
import { OptimizationEngine, SampleData } from "preference-opt-wasm";

/** Indicates the next action that should be done. */
export enum OptimizationState {
  Initial,
  Running,
}

export type TStore = {
  engine: OptimizationEngine;
  optState: OptimizationState;
  sampleData?: SampleData;
  bestSample?: Float64Array;
  savedSamples: Float64Array[];
  loadSampleData: () => void;
  selectSample: (idx: number) => void;
  saveCurrentBest: () => void;
};

const newEngine = () =>
  OptimizationEngine.new_empty(3).with_same_bounds(new Float64Array([0, 1]));

export const useStore = create<TStore>((set, get) => ({
  engine: newEngine(),
  optState: OptimizationState.Initial,
  savedSamples: [],
  loadSampleData: () => {
    const sampleData = get().engine.get_next_sample(
      500,
      1,
      get().sampleData?.f_prior()
    );
    set({ sampleData, optState: OptimizationState.Running });
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
      engine: newEngine(),
      bestSample: undefined,
    });
    get().loadSampleData();
  },
}));
