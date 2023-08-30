import { Link } from 'react-router-dom';

function NotFound() {
  return (
    <>
      <h1 className="text-white font-display">Not Found</h1>
      <Link to="/" className="text-white font-display">
        GO HOME
      </Link>
    </>
  );
}

export default NotFound;
