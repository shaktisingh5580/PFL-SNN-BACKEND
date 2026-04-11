"""Quick test for the Groq-powered LangGraph agent."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

print(f"GROQ_API_KEY set: {'Yes' if os.getenv('GROQ_API_KEY', '').startswith('gsk_') else 'No'}")

from src.chatbot.agent import GeospatialAgent

agent = GeospatialAgent()
print(f"Agent ready: {agent.is_ready}")

if agent.is_ready:
    print("\n--- Test 1: Show recent scans ---")
    result = agent.chat_sync("Show me all recent scans")
    print(result["response"][:600])
    
    print("\n--- Test 2: What violations were found? ---")
    result = agent.chat_sync("What violations were found?")
    print(result["response"][:600])
    
    print("\n--- Test 3: What are the compliance rules? ---")
    result = agent.chat_sync("What are the compliance rules configured in the system?")
    print(result["response"][:600])
else:
    print("\nAgent not ready. Testing fallback mode...")
    result = agent.chat_sync("Show me all recent scans")
    print(result["response"][:600])
