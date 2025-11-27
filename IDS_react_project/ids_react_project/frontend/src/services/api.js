// import axios from "axios";

// const API_URL = "http://localhost:5000/api";

// export const getLivePredictions = async () => {
//   try {
//     const res = await axios.get(`${API_URL}/prediction/live`);
//     return res.data;
//   } catch (error) {
//     console.error("Error fetching predictions:", error);
//     return [];
//   }
// };

// export const getDashboardStats = async () => {
//   try {
//     const res = await axios.get(`${API_URL}/dashboard`);
//     return res.data;
//   } catch (error) {
//     console.error("Error fetching dashboard:", error);
//     return { total_flows: 0, malicious_flows: 0, risk: "0%" };
//   }
// };




// import axios from "axios";

// const API_BASE = "http://localhost:5000/api";

// export const getLivePredictions = async () => {
//   try {
//     const res = await axios.get(`${API_BASE}/prediction/live`);
//     return res.data; // [{src_ip, dst_ip, pred_label}, ...]
//   } catch (err) {
//     console.error(err);
//     return [];
//   }
// };

// export const getDashboardStats = async () => {
//   try {
//     const res = await axios.get(`${API_BASE}/dashboard`);
//     return res.data; // { total_flows, malicious_flows, risk }
//   } catch (err) {
//     console.error(err);
//     return { total_flows: 0, malicious_flows: 0, risk: "0%" };
//   }
// };



import axios from "axios";

const BASE_URL = "http://localhost:5000/api";

export const getLivePredictions = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/prediction/live`);
    return response.data;
  } catch (err) {
    console.error("Error fetching live predictions:", err);
    return [];
  }
};

export const getDashboardData = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/dashboard`);
    return response.data;
  } catch (err) {
    console.error("Error fetching dashboard data:", err);
    return { total_packets: 0, malicious_flows: 0, risk_percent: 0 };
  }
};
