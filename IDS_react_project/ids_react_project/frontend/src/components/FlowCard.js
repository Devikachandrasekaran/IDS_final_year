// import React from "react";

// function FlowCard({ flow }) {
//   return (
//     <div className={`flow-card ${flow.pred === 1 ? "malicious" : "safe"}`}>
//       <p><strong>Source IP:</strong> {flow.src_ip}</p>
//       <p><strong>Destination IP:</strong> {flow.dst_ip}</p>
//       <p><strong>Status:</strong> {flow.pred === 1 ? "Malicious" : "Safe"}</p>
//     </div>
//   );
// }

// export default FlowCard;



// import React from "react";

// function FlowCard({ flow }) {
//   return (
//     <div className="flow-card">
//       <p>{flow.src_ip} -> {flow.dst_ip}</p>
//       <p>Predicted: {flow.pred_label}</p>
//     </div>
//   );
// }

// export default FlowCard;



// import React from "react";

// function FlowCard({ flow }) {
//   return (
//     <div className="flow-card">
//       <p>{flow?.src_ip} - {flow?.dst_ip}</p>
//       <p>Predicted: {flow?.pred_label}</p>
//     </div>
//   );
// }

// export default FlowCard;



import React from "react";

function FlowCard({ flow }) {
  return (
    <div className="flow-card">
      <p>{flow.src_ip} → {flow.dst_ip}</p>
      <p>Predicted: {flow.pred_label === 1 ? "Malicious" : "Safe"}</p>
    </div>
  );
}

export default FlowCard;
