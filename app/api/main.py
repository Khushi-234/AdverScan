from fastapi import FastAPI

app = FastAPI(
    title="AdverScan API",
    description="API for Adversarial Hardening and Testing Framework",
    version="0.1.0",
)

@app.get("/")
def read_root():
    return {
        "name": "AdverScan API",
        "status": "operational",
        "version": "0.1.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
