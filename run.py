"""
run.py
SceneIQ — Application entry point.

Initialises the Flask application and starts the development server.
No business logic lives here.
"""

from ui.app import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
