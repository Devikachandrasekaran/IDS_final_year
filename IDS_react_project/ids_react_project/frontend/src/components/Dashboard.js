// import React, { useEffect, useState } from "react";
// import { getDashboardStats } from "../services/api";
// import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

// function Dashboard() {
//   const [stats, setStats] = useState({ total_flows: 0, malicious_flows: 0, risk: "0%" });

//   useEffect(() => {
//     const interval = setInterval(async () => {
//       const data = await getDashboardStats();
//       setStats(data);
//     }, 3000);
//     return () => clearInterval(interval);
//   }, []);

//   const data = [
//     { name: "Malicious", value: stats.malicious_flows },
//     { name: "Safe", value: stats.total_flows - stats.malicious_flows },
//   ];

//   const COLORS = ["#ff4d4d", "#4caf50"];

//   return (
//     <div className="dashboard">
//       <h1>Dashboard</h1>
//       <div className="stats">
//         <p><strong>Total Packets:</strong> {stats.total_flows}</p>
//         <p><strong>Malicious Packets:</strong> {stats.malicious_flows}</p>
//         <p><strong>Risk Level:</strong> {stats.risk}</p>
//       </div>

//       <PieChart width={400} height={300}>
//         <Pie
//           data={data}
//           cx={200}
//           cy={150}
//           labelLine={false}
//           label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
//           outerRadius={120}
//           fill="#8884d8"
//           dataKey="value"
//         >
//           {data.map((entry, i) => (
//             <Cell key={i} fill={COLORS[i % COLORS.length]} />
//           ))}
//         </Pie>
//         <Tooltip />
//         <Legend />
//       </PieChart>
//     </div>
//   );
// }

// export default Dashboard;




// import React, { useContext } from "react";
// import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
// import { PredictionContext } from "../context/PredictionContext";

// function Dashboard() {
//   const { stats } = useContext(PredictionContext);

//   const data = [
//     { name: "Malicious", value: stats.malicious_flows },
//     { name: "Safe", value: stats.total_flows - stats.malicious_flows },
//   ];

//   const COLORS = ["#ff4d4d", "#4caf50"];

//   return (
//     <div className="dashboard">
//       <h1>Dashboard</h1>
//       <div className="stats">
//         <p><strong>Total Packets:</strong> {stats.total_flows}</p>
//         <p><strong>Malicious Packets:</strong> {stats.malicious_flows}</p>
//         <p><strong>Risk Level:</strong> {stats.risk}</p>
//       </div>

//       <PieChart width={400} height={300}>
//         <Pie
//           data={data}
//           cx={200}
//           cy={150}
//           labelLine={false}
//           label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
//           outerRadius={120}
//           fill="#8884d8"
//           dataKey="value"
//         >
//           {data.map((entry, i) => (
//             <Cell key={i} fill={COLORS[i % COLORS.length]} />
//           ))}
//         </Pie>
//         <Tooltip />
//         <Legend />
//       </PieChart>
//     </div>
//   );
// }

// export default Dashboard;


// import React, { useState, useEffect } from "react";
// import { getDashboardStats } from "../services/api";
// import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

// function Dashboard() {
//   // Default values to prevent undefined errors
//   const [stats, setStats] = useState({
//     total_flows: 0,
//     malicious_flows: 0,
//     risk: "0%"
//   });

//   useEffect(() => {
//     const fetchStats = async () => {
//       try {
//         const data = await getDashboardStats();
//         if (data) setStats(data);
//       } catch (err) {
//         console.error("Failed to fetch dashboard stats:", err);
//       }
//     };

//     // Fetch immediately, then every 3 seconds
//     fetchStats();
//     const interval = setInterval(fetchStats, 3000);
//     return () => clearInterval(interval);
//   }, []);

//   const data = [
//     { name: "Malicious", value: stats.malicious_flows || 0 },
//     { name: "Safe", value: (stats.total_flows || 0) - (stats.malicious_flows || 0) }
//   ];

//   const COLORS = ["#ff4d4d", "#4caf50"];

//   return (
//     <div className="dashboard">
//       <h1>Dashboard</h1>
//       <div className="stats">
//         <p><strong>Total Packets:</strong> {stats.total_flows || 0}</p>
//         <p><strong>Malicious Packets:</strong> {stats.malicious_flows || 0}</p>
//         <p><strong>Risk Level:</strong> {stats.risk || "0%"}</p>
//       </div>

//       <PieChart width={400} height={300}>
//         <Pie
//           data={data}
//           cx={200}
//           cy={150}
//           labelLine={false}
//           label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
//           outerRadius={120}
//           fill="#8884d8"
//           dataKey="value"
//         >
//           {data.map((entry, i) => (
//             <Cell key={i} fill={COLORS[i % COLORS.length]} />
//           ))}
//         </Pie>
//         <Tooltip />
//         <Legend />
//       </PieChart>
//     </div>
//   );
// }

// export default Dashboard;




// import React, { useEffect, useState } from 'react';
// import axios from 'axios';

// function Dashboard() {
//   const [liveData, setLiveData] = useState({
//     total_packets: 0,
//     malicious: 0,
//     benign: 0,
//     latest_flows: []
//   });

//   useEffect(() => {
//     const interval = setInterval(() => {
//       axios.get('http://localhost:5000/api/prediction/live')
//         .then(res => {
//           console.log("Live data:", res.data);
//           setLiveData(res.data);
//         })
//         .catch(err => {
//           console.error("Error fetching live data:", err);
//         });
//     }, 2000);

//     return () => clearInterval(interval);
//   }, []);

//   return (
//     <div className="p-6">
//       <h2 className="text-xl font-bold mb-4">Live Packet Prediction</h2>
//       <p>Total Packets: {liveData.total_packets}</p>
//       <p>Malicious: {liveData.malicious}</p>
//       <p>Benign: {liveData.benign}</p>

//       <h3 className="text-lg font-semibold mt-4">Latest Flows</h3>
//       <ul className="list-disc ml-6">
//         {liveData.latest_flows.map((item, idx) => (
//           <li key={idx}>
//             {item.flow} → Prediction: {item.prediction}
//           </li>
//         ))}
//       </ul>
//     </div>
//   );
// }

// export default Dashboard;



import React, { useContext } from "react";
import { PredictionContext } from "../context/PredictionContext";

function Dashboard() {
  const { dashboard } = useContext(PredictionContext);

  return (
    <div className="dashboard">
      <h1>Dashboard</h1>
      <p>Total Packets: {dashboard.total_packets}</p>
      <p>Malicious Packets: {dashboard.malicious_flows}</p>
      <p>Risk Level: {dashboard.risk_percent}%</p>
    </div>
  );
}

export default Dashboard;
