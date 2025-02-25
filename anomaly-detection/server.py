import json
import argparse
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler


class SensorDataHandler(BaseHTTPRequestHandler):
    """Handler for sensor data requests"""

    def __init__(self, output_dir, *args, **kwargs):
        self.output_dir = output_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Return server ready status"""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"1")  # Always ready to receive data

    def do_POST(self):
        """Handle incoming sensor data"""
        try:
            # Read and parse data
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            sensor_data = json.loads(post_data)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(self.output_dir) / f"sensor_data_{timestamp}.csv"

            # Save data to CSV
            self._save_data_to_csv(sensor_data, filepath)

            print(f"Data saved to {filepath}")
            self.send_response(204)  # Success, no content to return

        except Exception as e:
            print(f"Error processing data: {str(e)}")
            self.send_response(500)

        self.end_headers()

    def _save_data_to_csv(self, data, filepath):
        """Save sensor data to CSV file"""
        with open(filepath, "w") as f:
            num_samples = len(data["x"])
            for i in range(num_samples):
                f.write(f"{data['x'][i]},{data['y'][i]},{data['z'][i]}\n")


def create_server(output_dir, port):
    """Create and configure the HTTP server"""

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create handler with output directory configuration
    def handler(*args, **kwargs):
        return SensorDataHandler(output_dir, *args, **kwargs)

    # Create and return server
    return HTTPServer(("", port), handler)


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sensor Data Collection Server")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="sensor_data",
        help="Output directory for sensor data (default: sensor_data)",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=4242, help="Server port (default: 4242)"
    )
    args = parser.parse_args()

    # Create and start server
    server = create_server(args.dir, args.port)

    # Print startup message
    print("\nSensor Data Collection Server")
    print(f"Saving data to: {args.dir}")
    print(f"Server running on port {args.port}")
    print("Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
