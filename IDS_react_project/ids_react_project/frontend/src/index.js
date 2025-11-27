// import React from "react";
// import ReactDOM from "react-dom/client";
// import App from "./App";
// import "./styles.css";

// // Create the root React element and render the App component
// const root = ReactDOM.createRoot(document.getElementById("root"));
// root.render(
//   <React.StrictMode>
//     <App />
//   </React.StrictMode>
// );



// frontend/src/index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { PredictionProvider } from "./context/PredictionContext";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <PredictionProvider>
    <App />
  </PredictionProvider>
);
