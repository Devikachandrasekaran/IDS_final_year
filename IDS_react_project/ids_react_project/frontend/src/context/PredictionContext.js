// import React, { createContext, useState, useEffect } from "react";
// import { getLivePredictions, getDashboardStats } from "../services/api";

// export const PredictionContext = createContext();

// export function PredictionProvider({ children }) {
//   const [isRunning, setIsRunning] = useState(false);
//   const [flows, setFlows] = useState([]);
//   const [stats, setStats] = useState({ total_flows: 0, malicious_flows: 0, risk: "0%" });

//   // Poll live predictions
//   useEffect(() => {
//     let flowInterval;
//     if (isRunning) {
//       flowInterval = setInterval(async () => {
//         const data = await getLivePredictions();
//         setFlows(data);
//       }, 2000);
//     }
//     return () => clearInterval(flowInterval);
//   }, [isRunning]);

//   // Poll dashboard stats
//   useEffect(() => {
//     const statsInterval = setInterval(async () => {
//       const data = await getDashboardStats();
//       setStats(data);
//     }, 3000);
//     return () => clearInterval(statsInterval);
//   }, []);

//   return (
//     <PredictionContext.Provider value={{ isRunning, setIsRunning, flows, stats }}>
//       {children}
//     </PredictionContext.Provider>
//   );
// }

// frontend/src/context/PredictionContext.js
// import React, { createContext, useState, useEffect } from "react";
// import { getLivePredictions } from "../services/api";

// export const PredictionContext = createContext();

// export const PredictionProvider = ({ children }) => {
//   const [isRunning, setIsRunning] = useState(false);
//   const [flows, setFlows] = useState([]);

//   useEffect(() => {
//     let interval;

//     if (isRunning) {
//       interval = setInterval(async () => {
//         try {
//           const data = await getLivePredictions();
//           setFlows(data);
//         } catch (err) {
//           console.error("Error fetching live predictions:", err);
//         }
//       }, 2000);
//     }

//     return () => clearInterval(interval);
//   }, [isRunning]);

//   return (
//     <PredictionContext.Provider value={{ isRunning, setIsRunning, flows }}>
//       {children}
//     </PredictionContext.Provider>
//   );
// };



import React, { createContext, useState, useEffect } from "react";
import { getLivePredictions, getDashboardData } from "../services/api";

export const PredictionContext = createContext();

export const PredictionProvider = ({ children }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [flows, setFlows] = useState([]);
  const [dashboard, setDashboard] = useState({
    total_packets: 0,
    malicious_flows: 0,
    risk_percent: 0,
  });

  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(async () => {
        const liveFlows = await getLivePredictions();
        setFlows(liveFlows);

        const dash = await getDashboardData();
        setDashboard(dash);
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <PredictionContext.Provider
      value={{ isRunning, setIsRunning, flows, dashboard }}
    >
      {children}
    </PredictionContext.Provider>
  );
};
