# run_api.py
import uvicorn
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/api.log')
    ]
)

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info",
        access_log=True
    )
