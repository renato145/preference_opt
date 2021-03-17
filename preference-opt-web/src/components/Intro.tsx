import React, { HTMLProps } from "react";
import { TStore, useStore } from "../store";

const selector = ({ loadSampleData }: TStore) => loadSampleData;

export const Intro: React.FC<HTMLProps<HTMLDivElement>> = ({ ...props }) => {
  const loadSampleData = useStore(selector);
  return (
    <div {...props}>
      <button className="btn px-6 py-2" onClick={loadSampleData}>
        Start
      </button>
    </div>
  );
};
