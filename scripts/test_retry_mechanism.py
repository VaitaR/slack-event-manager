#!/usr/bin/env python3
"""Test retry mechanism for LLM client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import pytz
from src.adapters.llm_client import LLMClient
from src.config.settings import get_settings


def test_retry_with_short_timeout():
    """Test retry mechanism with very short timeout to trigger timeout errors."""
    print("🧪 Testing LLM retry mechanism with short timeout")
    print("=" * 70)
    
    settings = get_settings()
    
    # Create client with very short timeout (1s) to likely trigger timeout
    llm_client = LLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=1,  # Very short timeout to trigger retry
        verbose=False,
    )
    
    # Test with a long message that might timeout
    test_text = """
    <!here> Hi everyone! We've rolled out the new Deposit screen in Wallet! 
    This update completely rethinks how users add funds, making the experience 
    clearer, more personalized, and more conversion-friendly — especially for 
    users from Russia and nearby regions. What's new: Payment methods are now 
    grouped by fiat currency rather than KYC country. This means users immediately 
    see all relevant deposit options (like Apple Pay, cards, and P2P) that actually 
    work for their currency. Users can now easily select their preferred fiat 
    currency using the new fiat selection widget to view all available deposit 
    options in that currency. Automatic fiat selection based on user's KYC or IP.
    """ * 5  # Make it longer to increase processing time
    
    test_links = []
    test_ts = datetime.now(pytz.UTC)
    
    print(f"\n📝 Test message length: {len(test_text)} chars")
    print(f"⏱️ Timeout set to: 1 second (very short)")
    print(f"🔄 Max retries: 3")
    print("\nStarting test...\n")
    
    try:
        response = llm_client.extract_events_with_retry(
            text=test_text,
            links=test_links,
            message_ts_dt=test_ts,
            channel_name="test-channel",
            max_retries=3,
        )
        
        print("\n✅ Success! Response received:")
        print(f"   Is event: {response.is_event}")
        print(f"   Events: {len(response.events)}")
        
    except Exception as e:
        print(f"\n❌ Failed after all retries: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    print("\n" + "=" * 70)
    print("Test completed!")


def test_retry_with_normal_timeout():
    """Test with normal timeout (should work without retries)."""
    print("\n🧪 Testing LLM with normal timeout (30s)")
    print("=" * 70)
    
    settings = get_settings()
    
    # Create client with normal timeout
    llm_client = LLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=30,  # Normal timeout
        verbose=False,
    )
    
    test_text = "<!here> We've released a new feature for user authentication."
    test_links = []
    test_ts = datetime.now(pytz.UTC)
    
    print(f"\n📝 Test message length: {len(test_text)} chars")
    print(f"⏱️ Timeout set to: 30 seconds (normal)")
    print("\nStarting test...\n")
    
    try:
        response = llm_client.extract_events_with_retry(
            text=test_text,
            links=test_links,
            message_ts_dt=test_ts,
            channel_name="test-channel",
            max_retries=3,
        )
        
        print("\n✅ Success! Response received:")
        print(f"   Is event: {response.is_event}")
        print(f"   Events: {len(response.events)}")
        if response.events:
            for i, evt in enumerate(response.events, 1):
                print(f"   {i}. {evt.title}")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    print("\n" + "=" * 70)
    print("Test completed!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LLM RETRY MECHANISM TEST")
    print("=" * 70)
    
    # Test 1: Normal timeout (should succeed quickly)
    test_retry_with_normal_timeout()
    
    # Test 2: Short timeout (will likely trigger retries)
    # Uncomment to test retry mechanism:
    # test_retry_with_short_timeout()
    
    print("\n✅ All tests completed!\n")


