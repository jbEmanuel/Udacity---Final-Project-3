import json
from fastapi.testclient import TestClient

client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    print(r.status_code)
    
def test_post():
    data = json.dumps({
    'alcohol': 2.5,
    'malic_acid': 1.8, 
    'ash': 5.5,  
    'alcalinity_of_ash': 10.0        
    })
    r = client.post("https://ef6a-35-221-42-35.ngrok.io/predict/", data=data)
    print(r.json())