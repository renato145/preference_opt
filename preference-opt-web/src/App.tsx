import React from "react";
import { OptimizationEngine } from "preference-opt-wasm";

export const App = () => {
  const samples = new Float64Array([1, 0, 2, 4]);
  const preferences = new Uint32Array([0, 1]);
  const engine = OptimizationEngine.new(samples, preferences, 2);
  const dims = engine.dims();

  return (
    <div className="bg-blue-200 p-20">
      <p className="p-5 text-red-900">Dims: {dims}</p>
    </div>
  );
};
