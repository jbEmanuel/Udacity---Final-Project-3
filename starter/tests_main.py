import json
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# Write tests using the same syntax as with the requests module.
def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    print(r.status_code)
    
def test_post():
    data = json.dumps({
      "fnlgt": 122,
      "age": 25,
      "capital_gain": 12,
      "hours_per_week": 48,
      "civilian_spouse": "true",
      "education_num": 12,
      "relationship_Husband": "false",
      "capital_loss": 0
    })
    r = client.post("/predict/", data=data)
    print(r.json())
    

if __name__=="__main__":
  test_get_root()
  test_post()
  

    