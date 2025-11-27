// import React, { useState, useEffect } from "react";
// import { getLivePredictions } from "../services/api";
// import FlowCard from "./FlowCard";

// function Home() {
//   const [isRunning, setIsRunning] = useState(false);
//   const [flows, setFlows] = useState([]);

//   useEffect(() => {
//     let interval;
//     if (isRunning) {
//       interval = setInterval(async () => {
//         const data = await getLivePredictions();
//         setFlows(data);
//       }, 2000);
//     }
//     return () => clearInterval(interval);
//   }, [isRunning]);

//   return (
//     <div className="home">
//       <h1>Live Packet Prediction</h1>
//       <button
//         className={isRunning ? "stop-btn" : "start-btn"}
//         onClick={() => setIsRunning(!isRunning)}
//       >
//         {isRunning ? "Stop Prediction" : "Start Prediction"}
//       </button>

//       {flows.length === 0 ? (
//         <p>No live data yet. Click "Start Prediction" to begin.</p>
//       ) : (
//         <div className="flow-list">
//           {flows.map((flow, index) => (
//             <FlowCard key={index} flow={flow} />
//           ))}
//         </div>
//       )}
//     </div>
//   );
// }

// export default Home;


// import React, { useContext } from "react";
// import FlowCard from "./FlowCard";
// import { PredictionContext } from "../context/PredictionContext";

// function Home() {
//   const { isRunning, setIsRunning, flows } = useContext(PredictionContext);

//   return (
//     <div className="home">
//       <h1>Live Packet Prediction</h1>
//       <button
//         className={isRunning ? "stop-btn" : "start-btn"}
//         onClick={() => setIsRunning(!isRunning)}
//       >
//         {isRunning ? "Stop Prediction" : "Start Prediction"}
//       </button>

//       {flows.length === 0 ? (
//         <p>No live data yet. Click "Start Prediction" to begin.</p>
//       ) : (
//         <div className="flow-list">
//           {flows.map((flow, index) => (
//             <FlowCard key={index} flow={flow} />
//           ))}
//         </div>
//       )}
//     </div>
//   );
// }

// export default Home;



// import React, { useContext } from "react";
// import FlowCard from "./FlowCard";
// import { PredictionContext } from "../context/PredictionContext";

// function Home() {
//   const { isRunning, setIsRunning, flows } = useContext(PredictionContext);

//   return (
//     <div className="home">
//       <h1>Live Packet Prediction</h1>
//       <button
//         className={isRunning ? "stop-btn" : "start-btn"}
//         onClick={() => setIsRunning(!isRunning)}
//       >
//         {isRunning ? "Stop Prediction" : "Start Prediction"}
//       </button>

//       {flows.length === 0 ? (
//         <p>No live data yet. Click "Start Prediction" to begin.</p>
//       ) : (
//         <div className="flow-list">
//           {flows.map((flow, index) => (
//             <FlowCard key={index} flow={flow} />
//           ))}
//         </div>
//       )}
//     </div>
//   );
// }

// export default Home;


import React, { useContext } from "react";
import FlowCard from "./FlowCard";
import { PredictionContext } from "../context/PredictionContext";

function Home() {
  const { isRunning, setIsRunning, flows } = useContext(PredictionContext);

  return (
    <div className="home">
      <h1>Live Packet Prediction</h1>
      <button
        className={isRunning ? "stop-btn" : "start-btn"}
        onClick={() => setIsRunning(!isRunning)}
      >
        {isRunning ? "Stop Prediction" : "Start Prediction"}
      </button>

      {flows.length === 0 ? (
        <p>No live data yet. Click "Start Prediction" to begin.</p>
      ) : (
        <div className="flow-list">
          {flows.map((flow, index) => (
            <FlowCard key={index} flow={flow} />
          ))}
        </div>
      )}
    </div>
  );
}

export default Home;
