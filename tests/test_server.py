import pytest
from fastapi.testclient import TestClient
from visionbox.server import app, models, loading_states

client = TestClient(app)

def test_free_memory_endpoint():
    # Setup: Mock some data in the global dictionaries to simulate loaded models
    models["dummy_model_key"] = ("dummy_processor", "dummy_model_weights")
    loading_states["dummy_state_key"] = "done"

    # Verify they are in the dictionary
    assert "dummy_model_key" in models
    assert "dummy_state_key" in loading_states

    # Action: Call the free-memory API
    response = client.post("/api/free-memory")
    
    # Assert HTTP stats
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "GPU memory freed successfully."}

    # Assert dictionaries are cleared
    assert len(models) == 0
    assert len(loading_states) == 0
