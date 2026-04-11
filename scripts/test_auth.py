"""Quick auth test - raw HTTP to see exact error response."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("SH_CLIENT_ID")
client_secret = os.getenv("SH_CLIENT_SECRET")
token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

print(f"Client ID:     {client_id}")
print(f"Client Secret: {client_secret[:8]}...")
print(f"Token URL:     {token_url}")
print()

# Raw OAuth2 client_credentials request
response = requests.post(token_url, data={
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret,
})

print(f"HTTP Status: {response.status_code}")
print(f"Response:    {response.text[:500]}")

if response.status_code == 200:
    token = response.json().get("access_token", "")
    print(f"\n✅ AUTH SUCCESS! Token: {token[:30]}...")
else:
    print(f"\n❌ AUTH FAILED!")
    print(f"  This usually means:")
    print(f"  1. Client ID/Secret was copied with extra spaces")
    print(f"  2. The OAuth client was not created for 'client_credentials' grant type")
    print(f"  3. Check: https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings")
