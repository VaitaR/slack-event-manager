"""Interactive Telegram authentication script.

Creates session file for Telethon user client.
Run this script once before using Telegram integration.

Usage:
    python scripts/telegram_auth.py

Requirements:
    - TELEGRAM_API_ID in .env
    - TELEGRAM_API_HASH in .env
    - Phone number with Telegram account
    - Access to SMS/Telegram app for verification code
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from telethon import TelegramClient

# Load environment variables
load_dotenv()


async def authenticate() -> None:
    """Run interactive authentication flow."""
    # Get credentials from environment
    api_id_str = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    if not api_id_str or not api_hash:
        print("❌ Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        print("\nTo get credentials:")
        print("1. Go to https://my.telegram.org")
        print("2. Log in with your phone number")
        print("3. Go to 'API development tools'")
        print("4. Create an application")
        print("5. Copy API ID and API hash to .env")
        sys.exit(1)

    try:
        api_id = int(api_id_str)
    except ValueError:
        print(f"❌ Error: TELEGRAM_API_ID must be an integer, got: {api_id_str}")
        sys.exit(1)

    # Session file path
    session_path = "data/telegram_session"
    Path("data").mkdir(exist_ok=True)

    print("=" * 60)
    print("Telegram Authentication")
    print("=" * 60)
    print(f"API ID: {api_id}")
    print(f"Session file: {session_path}.session")
    print()

    # Create client
    client = TelegramClient(session_path, api_id, api_hash)

    try:
        print("🔐 Starting authentication...")
        print()

        # Start client (will prompt for phone and code)
        await client.start()

        # Get current user info
        me = await client.get_me()

        print()
        print("=" * 60)
        print("✅ Authentication successful!")
        print("=" * 60)
        print(f"User ID: {me.id}")
        print(f"Username: @{me.username}" if me.username else "Username: (not set)")
        print(f"First name: {me.first_name}")
        print(f"Last name: {me.last_name}" if me.last_name else "")
        print()
        print(f"Session saved to: {session_path}.session")
        print()
        print("You can now use Telegram integration in the pipeline.")
        print("Run: python scripts/run_multi_source_pipeline.py --source telegram")

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        sys.exit(1)

    finally:
        await client.disconnect()


def main() -> None:
    """Main entry point."""
    print()
    print("This script will authenticate your Telegram account.")
    print("You will need:")
    print("  - Your phone number (with country code, e.g., +1234567890)")
    print("  - Access to Telegram app or SMS for verification code")
    print()

    try:
        asyncio.run(authenticate())
    except KeyboardInterrupt:
        print("\n\n⚠️  Authentication cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
