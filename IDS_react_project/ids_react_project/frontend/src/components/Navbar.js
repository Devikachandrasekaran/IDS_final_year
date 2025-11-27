// import React from "react";
// import { Link } from "react-router-dom";

// function Navbar() {
//   return (
//     <nav className="navbar">
//       <h2>Malicious Packet Detector</h2>
//       <div>
//         <Link to="/">Home</Link>
//         <Link to="/dashboard">Dashboard</Link>
//       </div>
//     </nav>
//   );
// }

// export default Navbar;




import React from "react";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav>
      <ul>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/dashboard">Dashboard</Link></li>
      </ul>
    </nav>
  );
}

export default Navbar;
