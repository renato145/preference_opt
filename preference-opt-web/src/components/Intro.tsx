import React, { HTMLProps } from "react";
import { TStore, useStore } from "../store";

const selector = ({ loadSampleData }: TStore) => loadSampleData;

export const Intro: React.FC<HTMLProps<HTMLDivElement>> = ({ ...props }) => {
  const loadSampleData = useStore(selector);
  return (
    <div {...props}>
      <div className="mx-auto flex flex-col items-center">
        <p className="text-lg">
          Lorem ipsum dolor, sit amet consectetur adipisicing elit. Corporis
          labore pariatur molestias architecto illum impedit adipisci dolores!
          Totam ipsa magnam qui rem aliquid non dicta eum. Aspernatur at sequi
          sapiente!
        </p>
        <button
          className="mt-6 px-40 btn-confirm btn-lg"
          onClick={loadSampleData}
        >
          Start
        </button>
      </div>
    </div>
  );
};
