// import React from "react";
// import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
// import Navbar from "./components/Navbar";
// import Home from "./components/Home";
// import Dashboard from "./components/Dashboard";
// import "./styles.css";

// function App() {
//   return (
//     <Router>
//       <Navbar />
//       <div className="container">
//         <Routes>
//           <Route path="/" element={<Home />} />
//           <Route path="/dashboard" element={<Dashboard />} />
//         </Routes>
//       </div>
//     </Router>
//   );
// }

// export default App;




// import React from "react";
// import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
// import { PredictionProvider } from "./context/PredictionContext";
// import Home from "./components/Home";
// import Dashboard from "./components/Dashboard";
// import Navbar from "./components/Navbar";

// function App() {
//   return (
//     <PredictionProvider>
//       <Router>
//         <Navbar />
//         <Routes>
//           <Route path="/" element={<Home />} />
//           <Route path="/dashboard" element={<Dashboard />} />
//         </Routes>
//       </Router>
//     </PredictionProvider>
//   );
// }

// export default App;


import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { PredictionProvider } from "./context/PredictionContext";
import Home from "./components/Home";
import Dashboard from "./components/Dashboard";
import Navbar from "./components/Navbar";
import "./styles.css";

function App() {
  return (
    <PredictionProvider>
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </Router>
    </PredictionProvider>
  );
}

export default App;
