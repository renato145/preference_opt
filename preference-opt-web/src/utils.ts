export const array2color = (x: Float64Array) => {
  const sample = Array.from(x).map((o) => (o * 255).toFixed());
  return `rgb(${sample})`;
};
